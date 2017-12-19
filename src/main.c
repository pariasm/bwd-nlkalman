#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

// some macros and data types [[[1

//#define DUMP_INFO

// comment for a simpler version without keeping track of pixel variances
//#define VARIANCES

// comment for uniform aggregation
//#define WEIGHTED_AGGREGATION

#define max(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a > _b ? _a : _b; })

#define min(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

// read/write image sequence [[[1
static
float * vio_read_video_float_vec(const char * const path, int first, int last, 
		int *w, int *h, int *pd)
{
	// retrieve size from first frame and allocate memory for video
	int frames = last - first + 1;
	int whc;
	float *vid = NULL;
	{
		char frame_name[512];
		sprintf(frame_name, path, first);
		float *im = iio_read_image_float_vec(frame_name, w, h, pd);

		// size of a frame
		whc = *w**h**pd;

		vid = malloc(frames*whc*sizeof(float));
		memcpy(vid, im, whc*sizeof(float));
		if(im) free(im);
	}

	// load video
	for (int f = first + 1; f <= last; ++f)
	{
		int w1, h1, c1;
		char frame_name[512];
		sprintf(frame_name, path, f);
		float *im = iio_read_image_float_vec(frame_name, &w1, &h1, &c1);

		// check size
		if (whc != w1*h1*c1)
		{
			fprintf(stderr, "Size missmatch when reading frame %d\n", f);
			if (im)  free(im);
			if (vid) free(vid);
			return NULL;
		}

		// copy to video buffer
		memcpy(vid + (f - first)*whc, im, whc*sizeof(float));
		if(im) free(im);
	}

	return vid;
}

static
void vio_save_video_float_vec(const char * const path, float * vid, 
		int first, int last, int w, int h, int c)
{
	const int whc = w*h*c;
	for (int f = first; f <= last; ++f)
	{
		char frame_name[512];
		sprintf(frame_name, path, f);
		float * im = vid + (f - first)*whc;
		iio_save_image_float_vec(frame_name, im, w, h, c);
	}
}


// bicubic interpolation [[[1

#ifdef NAN
// extrapolate by nan
inline static
float getsample_nan(float *x, int w, int h, int pd, int i, int j, int l)
{
	assert(l >= 0 && l < pd);
	return (i < 0 || i >= w || j < 0 || j >= h) ? NAN : x[(i + j*w)*pd + l];
}
#endif//NAN

inline static
float cubic_interpolation(float v[4], float x)
{
	return v[1] + 0.5 * x*(v[2] - v[0]
			+ x*(2.0*v[0] - 5.0*v[1] + 4.0*v[2] - v[3]
			+ x*(3.0*(v[1] - v[2]) + v[3] - v[0])));
}

static
float bicubic_interpolation_cell(float p[4][4], float x, float y)
{
	float v[4];
	v[0] = cubic_interpolation(p[0], y);
	v[1] = cubic_interpolation(p[1], y);
	v[2] = cubic_interpolation(p[2], y);
	v[3] = cubic_interpolation(p[3], y);
	return cubic_interpolation(v, x);
}

static
void bicubic_interpolation_nans(float *result,
		float *img, int w, int h, int pd, float x, float y)
{
	x -= 1;
	y -= 1;

	int ix = floor(x);
	int iy = floor(y);
	for (int l = 0; l < pd; l++) {
		float c[4][4];
		for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
			c[i][j] = getsample_nan(img, w, h, pd, ix + i, iy + j, l);
		float r = bicubic_interpolation_cell(c, x - ix, y - iy);
		result[l] = r;
	}
}

static
void warp_bicubic_inplace(float *imw, float *im, float *of, int w, int h, int ch)
{
	// warp previous frame
	for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
		float xw = x + of[(x + y*w)*2 + 0];
		float yw = y + of[(x + y*w)*2 + 1];
		bicubic_interpolation_nans(imw + (x + y*w)*ch, im, w, h, ch, xw, yw);
	}
	return;
}

/* static
float * warp_bicubic(float * im, float * of, int w, int h, int ch)
{
	// warp previous frame
	float * im_w = malloc(w*h*ch * sizeof(float));
	warp_bicubic_inplace(im_w, im, of, w, h, ch);
	return im_w;
} */

// dct handler [[[1

// dct implementation: using fftw or as a matrix product
enum dct_method {FFTW, MATPROD};

// struct containing the workspaces for the dcts
struct dct_threads
{
	// workspaces for DCT transforms in each thread
	fftwf_plan plan_forward [100];
	fftwf_plan plan_backward[100];
	float *dataspace[100];
	float *datafreq [100];

	// size of the signals
	int width;
	int height;
	int frames;
	int nsignals;
	int nthreads;

	// DCT bases for matrix product implementation 
	// TODO
	
	// which implementation to use
	enum dct_method method;
};

// init dct workspaces [[[2
void dct_threads_init(int w, int h, int f, int n, int t, struct dct_threads * dct_t)
{
#ifdef _OPENMP
	t = min(t, omp_get_max_threads());
	if (t > 100)
	{
		fprintf(stderr,"Error: dct_threads is hard-coded"
		               "for a maximum of 100 threads\n");
		exit(1);
	}
#else
	if (t > 1)
	{
		fprintf(stderr,"Error: dct_threads can't handle"
		               "%d threads (no OpenMP)\n", t);
		exit(1);
	}
#endif

	dct_t->width = w;
	dct_t->height = h;
	dct_t->frames = f;
	dct_t->nsignals = n;
	dct_t->nthreads = t;

   unsigned int N = w * h * f * n;

	// define method based on patch size
//	dct_t->method = (width * height * frames < 32) ? MATPROD : FFTW;
	dct_t->method = FFTW;
//	dct_t->method = MATPROD; // FIXME: MATPROD IS NOT WORKING!

//	fprintf(stderr, "init DCT for %d thread - %d x %d x %d ~ %d\n",nthreads, width, height, frames, nsignals);

	switch (dct_t->method)
	{
		case FFTW:

			for (int i = 0; i < dct_t->nthreads; i++)
			{
				dct_t->dataspace[i] = (float*)fftwf_malloc(sizeof(float) * N);
				dct_t->datafreq [i] = (float*)fftwf_malloc(sizeof(float) * N);
		
				int sz[] = {f, h, w};
		
				fftwf_r2r_kind dct[] = {FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10};
				dct_t->plan_forward[i] = fftwf_plan_many_r2r(3, sz, n,
						dct_t->dataspace[i], NULL, 1, w * h * f, 
						dct_t->datafreq [i], NULL, 1, w * h * f,
//						dct_t->dataspace[i], NULL, n, 1, 
//						dct_t->datafreq [i], NULL, n, 1,
						dct, FFTW_ESTIMATE);
		
				fftwf_r2r_kind idct[] = {FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01};
				dct_t->plan_backward[i] = fftwf_plan_many_r2r(3, sz, n,
						dct_t->datafreq [i], NULL, 1, w * h * f,
						dct_t->dataspace[i], NULL, 1, w * h * f,
//						dct_t->datafreq [i], NULL, n, 1,
//						dct_t->dataspace[i], NULL, n, 1,
						idct, FFTW_ESTIMATE);
			}
			break;
			
		case MATPROD:

			fprintf(stderr, "MATPROD DCT HANDLER IS NOT YET IMPLEMENTED\n");
			break;
	}

}

// delete dct workspaces [[[2
void dct_threads_destroy(struct dct_threads * dct_t)
{
	if (dct_t->nsignals)
	{
		for (int i = 0; i < dct_t->nthreads; i++)
		{
			fftwf_free(dct_t->dataspace[i]);
			fftwf_free(dct_t->datafreq [i]);
			fftwf_destroy_plan(dct_t->plan_forward [i]);
			fftwf_destroy_plan(dct_t->plan_backward[i]);
		}
	}
}

// compute forward dcts [[[2
void dct_threads_forward(float * patch, struct dct_threads * dct_t)
{
   int tid = 0;
#ifdef _OPENMP
   tid = omp_get_thread_num();
#endif

	int w   = dct_t->width;
	int h   = dct_t->height;
	int f   = dct_t->frames;
	int n   = dct_t->nsignals;
	int wh  = w * h;
	int whf = wh * f;
	int N   = whf * n;

	if (N == 0)
		fprintf(stderr, "Attempting to use an uninitialized dct_threads struct.\n");

	switch (dct_t->method)
	{
		case MATPROD: // compute dct via separable matrix products

			fprintf(stderr, "MATPROD DCT HANDLER IS NOT YET IMPLEMENTED\n");
			break;

		case FFTW: // compute dct via fftw
		{
			// copy and compute unnormalized dct
			for (int i = 0; i < N; i++) dct_t->dataspace[tid][i] = patch[i];

			fftwf_execute(dct_t->plan_forward[tid]);

			// copy and orthonormalize
			float norm   = 1.0/sqrt(8.0*(float)whf);
			float isqrt2 = 1.f/sqrt(2.0);

			for (int i = 0; i < N; i++) patch[i] = dct_t->datafreq[tid][i] * norm;

			for (int i = 0; i < n; i++)
			{
				for (int t = 0; t < f; t++)
				for (int y = 0; y < h; y++)
					patch[i*whf + t*wh + y*w] *= isqrt2;

				for (int t = 0; t < f; t++)
				for (int x = 0; x < w; x++)
					patch[i*whf + t*wh + x] *= isqrt2;

				for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					patch[i*whf + y*w + x] *= isqrt2;
			}

			break;
		}
	}
}

// compute inverse dcts [[[2
void dct_threads_inverse(float * patch, struct dct_threads * dct_t)
{
	int tid = 0;
#ifdef _OPENMP
   tid = omp_get_thread_num();
#endif

	int w   = dct_t->width;
	int h   = dct_t->height;
	int f   = dct_t->frames;
	int n   = dct_t->nsignals;
	int wh  = w * h;
	int whf = wh * f;
	int N   = whf * n;

	if (N == 0)
		fprintf(stderr, "Attempting to use a uninitialized dct_threads struct.\n");

	switch (dct_t->method)
	{
		case MATPROD: // compute dct via separable matrix products

			fprintf(stderr, "MATPROD DCT HANDLER IS NOT YET IMPLEMENTED\n");
			break;

		case FFTW: // compute dct via fftw
		{
			// normalize
			float norm  = 1.0/sqrt(8.0*(float)whf);
			float sqrt2 = sqrt(2.0);

			for (int i = 0; i < n; i++)
			{
				for (int t = 0; t < f; t++)
				for (int y = 0; y < h; y++)
					patch[i*whf + t*wh + y*w] *= sqrt2;

				for (int t = 0; t < f; t++)
				for (int x = 0; x < w; x++)
					patch[i*whf + t*wh + x] *= sqrt2;

				for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					patch[i*whf + y*w + x] *= sqrt2;
			}

			for (int i = 0; i < N; i++) dct_t->datafreq[tid][i] = patch[i] * norm;

			fftwf_execute(dct_t->plan_backward[tid]);

			for (int i=0; i < N; i++) patch[i] = dct_t->dataspace[tid][i];
		}
	}
}


// recursive nl-means algorithm [[[1

// struct for storing the parameters of the algorithm
struct vnlmeans_params
{
	int patch_sz;        // patch size
	int search_sz;       // search window radius
	float dista_th;      // patch distance threshold
	float dista_lambda;  // weight of current frame in patch distance
	float beta_x;        // noise multiplier in spatial filtering
	float beta_t;        // noise multiplier in kalman filtering
	bool pixelwise;      // toggle pixel-wise nlmeans
};

// set default parameters as a function of sigma
void vnlmeans_default_params(struct vnlmeans_params * p, float sigma)
{
	const bool a = !(p->pixelwise); // set by caller
	if (p->patch_sz     < 0) p->patch_sz     = a ? 8 : 5;
	if (p->search_sz    < 0) p->search_sz    = 10;
	if (p->dista_th     < 0) p->dista_th     = 0.85 * sigma;
	if (p->dista_lambda < 0) p->dista_lambda = 1.;
	if (p->beta_x       < 0) p->beta_x       = 1;
	if (p->beta_t       < 0) p->beta_t       = 4;
}

// denoise frame t
void vnlmeans_frame(float *deno1, float *nisy1, float *deno0, 
		int w, int h, int ch, float sigma,
		const struct vnlmeans_params prms)
{
	// definitions [[[2

	const int psz = prms.patch_sz;
	const int step = prms.pixelwise ? 1 : psz/2;
//	const int step = prms.pixelwise ? 1 : psz;
	const float sigma2 = sigma * sigma;
	const float dista_th2 = prms.dista_th * prms.dista_th;
	const float beta_x  = prms.beta_x;
	const float beta_t  = prms.beta_t;

	// aggregation weights (not necessary for pixel-wise nlmeans)
	float *aggr1 = prms.pixelwise ? NULL : malloc(w*h*sizeof(float));

	// set output and aggregation weights to 0
	for (int i = 0; i < w*h*ch; ++i) deno1[i] = 0.;
	if (aggr1) for (int i = 0; i < w*h; ++i) aggr1[i] = 0.;

	// noisy and clean patches at point p (as VLAs in the stack!)
	float N1[psz][psz][ch]; // noisy patch at position p in frame t
	float D0[psz][psz][ch]; // denoised patch at p in frame t - 1

	// wrap images with nice pointers to vlas
	float (*a1)[w]     = (void *)aggr1;       // aggregation weights at t
	float (*d1)[w][ch] = (void *)deno1;       // denoised frame t (output)
	const float (*d0)[w][ch] = (void *)deno0; // denoised frame t-1
	const float (*n1)[w][ch] = (void *)nisy1; // noisy frame at t

	// initialize dct workspaces (we will compute the dct of two patches)
	float N1D0[2*ch][psz][psz]; // noisy patch at t and clean patch at t-1
	struct dct_threads dcts[1];
#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
#else
	const int nthreads = 1;
#endif
	dct_threads_init(psz, psz, 1, 2*ch, nthreads, dcts); // 2D DCT
//	dct_threads_init(psz, psz, 2, 1*ch, nthreads, dcts); // 3D DCT

	// statistics
	float M0 [ch][psz][psz]; // average patch at t-1
	float V0 [ch][psz][psz]; // variance at t-1
	float V01[ch][psz][psz]; // transition variance from t-1 to t
	float M1 [ch][psz][psz]; // average patch at t
	float V1 [ch][psz][psz]; // variance at t

	// loop on image patches [[[2
	for (int oy = 0; oy < psz; oy += step) // split in grids of non-overlapping
	for (int ox = 0; ox < psz; ox += step) // patches (for parallelization)
	#pragma omp parallel for private(N1D0,N1,D0,M0,V0,V01,M1,V1)
	for (int py = oy; py < h - psz + 1; py += psz) // FIXME: boundary pixels
	for (int px = ox; px < w - psz + 1; px += psz) // may not be denoised
	{
		//	load target patch [[[3
		bool prev_p = d0;
		for (int hy = 0; hy < psz; ++hy)
		for (int hx = 0; hx < psz; ++hx)
		{
			if (prev_p && isnan(d0[py + hy][px + hx][0])) prev_p = false;
			for (int c  = 0; c  < ch ; ++c )
			{
				D0[hy][hx][c] = (prev_p) ? d0[py + hy][px + hx][c] : 0.f;
				N1[hy][hx][c] = n1[py + hy][px + hx][c];

				M1 [c][hy][hx] = 0.;
				V1 [c][hy][hx] = 0.;
				M0 [c][hy][hx] = 0.;
				V0 [c][hy][hx] = 0.;
				V01[c][hy][hx] = 0.;
			}
		}

		// gather spatio-temporal statistics: loop on search region [[[3
		int np0 = 0; // number of similar patches with a  valid previous patch
		int np1 = 0; // number of similar patches with no valid previous patch
		if (dista_th2)
		{
			const int wsz = prms.search_sz;
			const int wx[2] = {max(px - wsz, 0), min(px + wsz, w - psz) + 1};
			const int wy[2] = {max(py - wsz, 0), min(py + wsz, h - psz) + 1};
			for (int qy = wy[0]; qy < wy[1]; ++qy)
			for (int qx = wx[0]; qx < wx[1]; ++qx)
			{
				// store patch at q [[[4

				// check if the previous patch at q is valid
				bool prev_q = d0;
				if (prev_q)
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					if (prev_q && isnan(d0[qy + hy][qx + hx][0]))
						prev_q = false;

				const bool prev = prev_p && prev_q;

				for (int c  = 0; c  < ch ; ++c )
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					N1D0[c     ][hy][hx] =        n1[qy + hy][qx + hx][c];
					N1D0[c + ch][hy][hx] = prev ? d0[qy + hy][qx + hx][c] : 0;
				}

				// compute patch distance [[[4
				float ww = 0; // patch distance is saved here
				const float l = prms.dista_lambda;
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
					if (prev && l != 1)
						// use noisy and denoised patches from previous frame
						for (int c  = 0; c  < ch ; ++c )
						{
							const float e1 = N1D0[c     ][hy][hx] - N1[hy][hx][c];
							const float e0 = N1D0[c + ch][hy][hx] - D0[hy][hx][c];
							ww += l * max(e1 * e1 - 2*sigma2,0) + (1 - l) * e0 * e0;
						}
					else
					{
						// use only noisy from current frame
						for (int c  = 0; c  < ch ; ++c )
						{
							const float e1 = N1D0[c][hy][hx] - N1[hy][hx][c];
							ww += max(e1 * e1 - 2*sigma2, 0);
						}
					}

				// normalize distance by number of pixels in patch
				ww /= (float)psz*psz*ch;

				// if patch at q is similar to patch at p, update statistics [[[4
				if (ww <= dista_th2)
				{
					np1++;
					np0 += prev ? 1 : 0;

					// compute dct (output in N1D0)
					dct_threads_forward((float *)N1D0, dcts);

					// compute means and variances.
					// to compute the variances in a single pass over the search
					// region we use Welford's method.
					const float inp0 = prev ? 1./(float)np0 : 0;
					const float inp1 = 1./(float)np1;
					for (int c  = 0; c  < ch ; ++c )
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					{
						const float p = N1D0[c][hy][hx];
						const float oldM1 = M1[c][hy][hx];
						const float delta = p - oldM1;

						M1[c][hy][hx] += delta * inp1;
						V1[c][hy][hx] += delta * (p - M1[c][hy][hx]); 

						if(prev)
						{
							float p = N1D0[c + ch][hy][hx];
							const float oldM0 = M0[c][hy][hx];
							const float delta = p - oldM0;

							M0[c][hy][hx] += delta * inp0;
							V0[c][hy][hx] += delta * (p - M0[c][hy][hx]);

							p -= N1D0[c][hy][hx];
							V01[c][hy][hx] += p*p;
						}
					}
				} // ]]]4
			}

			// correct variance [[[4
			const float inp0 = np0 ? 1./(float)np0 : 0;
			const float inp1 = 1./(float)np1;
			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				V1[c][hy][hx] *= inp1;
				if(np0)
				{
					V0 [c][hy][hx] *= inp0;
					V01[c][hy][hx] *= inp0;
				}
			}
			// ]]]4
		}
		else // dista_th2 == 0
		{
			// local version: single point estimate of variances [[[4
			//                the mean M1 is assumed to be 0

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				N1D0[c     ][hy][hx] =          N1[hy][hx][c];
				N1D0[c + ch][hy][hx] = prev_p ? D0[hy][hx][c] : 0;
			}

			// compute dct (output in N1D0)
			dct_threads_forward((float *)N1D0, dcts);

			// patch statistics (point estimate)
			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				float p = N1D0[c][hy][hx];
				V1[c][hy][hx] = p * p;

				if (prev_p)
				{
					p = N1D0[c + ch][hy][hx];
					V0[c][hy][hx] = p * p;

					p -= N1D0[c + ch][hy][hx];
					V01[c][hy][hx] = p * p;
				}
			}//]]]4
		}

		// filter current patch [[[3

		// load patch in memory for fftw
		for (int c  = 0; c  < ch ; ++c )
		for (int hy = 0; hy < psz; ++hy)
		for (int hx = 0; hx < psz; ++hx)
		{
			N1D0[c     ][hy][hx] =          N1[hy][hx][c];
			N1D0[c + ch][hy][hx] = prev_p ? D0[hy][hx][c] : 0;
		}

		// compute dct (computed in place in N1D0)
		dct_threads_forward((float *)N1D0, dcts);

		float vp = 0;
		if (np0 > 4) // enough patches with a valid previous patch
		{
			// "kalman"-like spatio-temporal denoising

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				// prediction variance (substract sigma2 from transition variance)
				float v = V0[c][hy][hx] + max(0.f, V01[c][hy][hx] - sigma2);

				// kalman gain
				float a = v / (v + beta_t * sigma2);
				if (a < 0) printf("a = %f v = %f ", a, v);
				if (a > 1) printf("a = %f v = %f ", a, v);

				// variance of filtered patch
				vp += (1 - a * a) * v - a * a * sigma2;

				// filter
				N1D0[c][hy][hx] = a*N1D0[c][hy][hx] + (1 - a)*N1D0[c + ch][hy][hx];
			}
		}
		else // not enough patches with valid previous patch
		{
			// spatial nl-dct using statistics in M1 V1

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				// prediction variance (substract sigma2 from transition variance)
				float v = max(0.f, V1[c][hy][hx] - sigma2);

				// wiener filter
				float a = v / (v + beta_x * sigma2);
				if (a < 0) printf("a = %f v = %f ", a, v);
				if (a > 1) printf("a = %f v = %f ", a, v);

				// variance of filtered patch
				vp += a * a * v;

				/* thresholding instead of empirical Wiener filtering
				vp += 1;

				float a = (hy != 0 || hx != 0) ?
					(N1D0[c][hy][hx] * N1D0[c][hy][hx] > 3 * sigma2) : 1;*/

				// filter
				N1D0[c][hy][hx] = a*N1D0[c][hy][hx] + (1 - a)*M1[c][hy][hx];

			}
		}

		// invert dct (output in N1D0)
		dct_threads_inverse((float *)N1D0, dcts);

		// aggregate denoised patch on output image [[[3
		if (a1)
		{
#ifdef WEIGHTED_AGGREGATION
			const float w = 1.f/vp;
#else
			const float w = 1.f;
#endif
			// patch-wise denoising: aggregate the whole denoised patch
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				a1[py + hy][px + hx] += w;
				for (int c = 0; c < ch ; ++c )
					d1[py + hy][px + hx][c] += w * N1D0[c][hy][hx];
			}
		}
		else 
			// pixel-wise denoising: aggregate only the central pixel
			for (int c = 0; c < ch ; ++c )
				d1[py + psz/2][px + psz/2][c] += N1D0[c][psz/2][psz/2];

		// ]]]3
	}

	// normalize output [[[2
	if (aggr1)
	for (int i = 0, j = 0; i < w*h; ++i) 
	for (int c = 0; c < ch ; ++c, ++j) 
		deno1[j] /= aggr1[i];

	// free allocated mem and quit
	dct_threads_destroy(dcts);
	if (aggr1) free(aggr1);

	return; // ]]]2
}

// main funcion [[[1

// 'usage' message in the command line
static const char *const usages[] = {
	"vnlmeans [options] [[--] args]",
	"vnlmeans [options]",
	NULL,
};

int main(int argc, const char *argv[])
{
	// parse command line [[[2

	// command line parameters and their defaults
	const char *nisy_path = NULL;
	const char *deno_path = NULL;
	const char *flow_path = NULL;
	int fframe = 0, lframe = -1;
	float sigma = 0.f;
	bool verbose = false;
	struct vnlmeans_params prms;
	prms.patch_sz     = -1; // -1 means automatic value
	prms.search_sz    = -1;
	prms.dista_th     = -1.;
	prms.beta_x       = -1.;
	prms.beta_t       = -1.;
	prms.dista_lambda = -1.;
	prms.pixelwise = false;

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Algorithm options"),
		OPT_STRING ('i', "nisy"  , &nisy_path, "noisy input path (printf format)"),
		OPT_STRING ('o', "flow"  , &flow_path, "backward flow path (printf format)"),
		OPT_STRING ('d', "deno"  , &deno_path, "denoised output path (printf format)"),
		OPT_INTEGER('f', "first" , &fframe, "first frame"),
		OPT_INTEGER('l', "last"  , &lframe , "last frame"),
		OPT_FLOAT  ('s', "sigma" , &sigma, "noise standard dev"),
		OPT_INTEGER('p', "patch" , &prms.patch_sz, "patch size"),
		OPT_INTEGER('w', "search", &prms.search_sz, "search region radius"),
		OPT_FLOAT  ( 0 , "dth"   , &prms.dista_th, "patch distance threshold"),
		OPT_FLOAT  ( 0 , "beta_x", &prms.beta_x, "noise multiplier in spatial filtering"),
		OPT_FLOAT  ( 0 , "beta_t", &prms.beta_t, "noise multiplier in kalman filtering"),
		OPT_FLOAT  ( 0 , "lambda", &prms.dista_lambda, "noisy patch weight in patch distance"),
		OPT_BOOLEAN( 0 , "pixel" , &prms.pixelwise, "toggle pixel-wise denoising"),
		OPT_GROUP("Program options"),
		OPT_BOOLEAN('v', "verbose", &verbose, "verbose output"),
		OPT_END(),
	};

	// parse command line
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nA video denoiser based on non-local means.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// default value for noise-dependent params
	vnlmeans_default_params(&prms, sigma);

	// print parameters
	if (verbose)
	{
		printf("parameters:\n");
		printf("\tnoise  %f\n", sigma);
		printf("\t%s-wise mode\n", prms.pixelwise ? "pixel" : "patch");
		printf("\tpatch     %d\n", prms.patch_sz);
		printf("\tsearch    %d\n", prms.search_sz);
		printf("\tdth       %g\n", prms.dista_th);
		printf("\tlambda    %g\n", prms.dista_lambda);
		printf("\tbeta_x    %g\n", prms.beta_x);
		printf("\tbeta_t    %g\n", prms.beta_t);
		printf("\n");
#ifdef VARIANCES
		printf("\tVARIANCES ON\n");
#endif
#ifdef WEIGHTED_AGGREGATION
		printf("\tWEIGHTED_AGGREGATION ON\n");
#endif
	}

	// load data [[[2
	if (verbose) printf("loading video %s\n", nisy_path);
	int w, h, c; //, frames = lframe - fframe + 1;
	float * nisy = vio_read_video_float_vec(nisy_path, fframe, lframe, &w, &h, &c);
	{
		if (!nisy)
			return EXIT_FAILURE;
	}

	// load optical flow
	float * flow = NULL;
	if (flow_path)
	{
		if (verbose) printf("loading flow %s\n", flow_path);
		int w1, h1, c1;
		flow = vio_read_video_float_vec(flow_path, fframe, lframe, &w1, &h1, &c1);

		if (!flow)
		{
			if (nisy) free(nisy);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 2)
		{
			fprintf(stderr, "Video and optical flow size missmatch\n");
			if (nisy) free(nisy);
			if (flow) free(flow);
			return EXIT_FAILURE;
		}
	}

	// run denoiser [[[2
	const int whc = w*h*c, wh2 = w*h*2;
	float * deno = nisy;
	float * warp0 = malloc(whc*sizeof(float));
	float * deno1 = malloc(whc*sizeof(float));
#ifdef VARIANCES
	float * vari0 = malloc(w*h*sizeof(float));
	float * vari1 = malloc(w*h*sizeof(float));
#endif
	for (int f = fframe; f <= lframe; ++f)
	{
		if (verbose) printf("processing frame %d\n", f);

		// TODO compute optical flow if absent

		// warp previous denoised frame
		if (f > fframe)
		{
			float * deno0 = deno + (f - 1 - fframe)*whc;
			if (flow)
			{
				float * flow0 = flow + (f - 0 - fframe)*wh2;
				warp_bicubic_inplace(warp0, deno0, flow0, w, h, c);
#ifdef VARIANCES
				warp_bicubic_inplace(vari0, vari1, flow0, w, h, 1);
#endif
			}
			else
				// copy without warping
				memcpy(warp0, deno0, whc*sizeof(float));
		}

		// run denoising
		float *nisy1 = nisy + (f - fframe)*whc;
		float *deno0 = (f > fframe) ? warp0 : NULL;
		vnlmeans_frame(deno1, nisy1, deno0, w, h, c, sigma, prms);
		memcpy(nisy1, deno1, whc*sizeof(float));
	}

	// save output [[[2
	vio_save_video_float_vec(deno_path, deno, fframe, lframe, w, h, c);

	if (deno1) free(deno1);
	if (warp0) free(warp0);
#ifdef VARIANCES
	if (vari0) free(vari0);
	if (vari1) free(vari1);
#endif
	if (nisy) free(nisy);
	if (flow) free(flow);

	return EXIT_SUCCESS; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
