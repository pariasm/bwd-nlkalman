#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

#include <stdio.h>     // getchar() for debugging

//#define DEBUG_OUTPUT_FILTERING
//#define DEBUG_OUTPUT_SMOOTHING

// some macros and data types [[[1

// comment for uniform aggregation
#define WEIGHTED_AGGREGATION

// comment for distance threshold
#define K_SIMILAR_PATCHES

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
void warp_bicubic(float *imw, float *im, float *of, float *msk,
		int w, int h, int ch)
{
	// warp previous frame
	for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	if (!msk || (msk &&  msk[x + y*w] == 0))
	{
		float xw = x + of[(x + y*w)*2 + 0];
		float yw = y + of[(x + y*w)*2 + 1];
		bicubic_interpolation_nans(imw + (x + y*w)*ch, im, w, h, ch, xw, yw);
	}
	else
		for (int c = 0; c < ch; ++c)
			imw[(x + y*w)*ch + c] = NAN;

	return;
}

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


// window functions [[[1

float * window_function(const char * type, int NN)
{
	const float N = (float)NN;
	const float N2 = (N - 1.)/2.;
	const float PI = 3.14159265358979323846;
	float w1[NN];

	if (strcmp(type, "parzen") == 0)
		for (int n = 0; n < NN; ++n)
		{
			float nc = (float)n - N2;
			w1[n] = (fabs(nc) <= N/4.)
			      ? 1. - 24.*nc*nc/N/N*(1. - 2./N*fabs(nc))
					: 2.*pow(1. - 2./N*fabs(nc), 3.);
		}
	else if (strcmp(type, "welch") == 0)
		for (int n = 0; n < NN; ++n)
		{
			const float nc = ((float)n - N2)/N2;
			w1[n] = 1. - nc*nc;
		}
	else if (strcmp(type, "sine") == 0)
		for (int n = 0; n < NN; ++n)
			w1[n] = sin(PI*(float)n/(N-1));
	else if (strcmp(type, "hanning") == 0)
		for (int n = 0; n < NN; ++n)
		{
			w1[n] = sin(PI*(float)n/(N-1));
			w1[n] *= w1[n];
		}
	else if (strcmp(type, "hamming") == 0)
		for (int n = 0; n < NN; ++n)
			w1[n] = 0.54 - 0.46*cos(2*PI*(float)n/(N-1));
	else if (strcmp(type, "blackman") == 0)
		for (int n = 0; n < NN; ++n)
			w1[n] = 0.42 - 0.5*cos(2*PI*(float)n/(N-1)) + 0.08*cos(4*PI*n/(N-1));
	else if (strcmp(type, "gaussian") == 0)
		for (int n = 0; n < NN; ++n)
		{
			const float s = .4; // scale parameter for the Gaussian
			const float x = ((float)n - N2)/N2/s;
			w1[n] = exp(-.5*x*x);
		}
	else // default is the flat window
		for (int n = 0; n < NN; ++n)
			w1[n] = 1.f;

	// 2D separable window
	float * w2 = malloc(NN*NN*sizeof(float));
	for (int i = 0; i < NN; ++i)
	for (int j = 0; j < NN; ++j)
		w2[i*NN + j] = w1[i]*w1[j];

	return w2;
}




// nl-kalman filtering [[[1

// struct for storing the parameters of the algorithm
struct nlkalman_params
{
	int patch_sz;        // patch size
	int search_sz;       // search window radius
#ifdef K_SIMILAR_PATCHES
	int num_patches_x;     // number of similar patches spatial filtering
	int num_patches_t;     // number of similar patches temporal filtering
	int num_patches_tx;    // number of similar patches statial average in temporal filtering
#else
	float dista_th;      // patch distance threshold
#endif
	float dista_lambda;  // weight of current frame in patch distance
	float beta_x;        // noise multiplier in spatial filtering
	float beta_t;        // noise multiplier in kalman filtering
	bool pixelwise;      // toggle pixel-wise nlmeans
};

// set default parameters as a function of sigma
void nlkalman_default_params(struct nlkalman_params * p, float sigma)
{
	/* we trained using two different datasets (both grayscale): 
	 * - derfhd: videos of half hd resolution obtained by downsampling
	 *           some hd videos from the derf database
	 * - derfcif: videos of cif resolution also of the derf db
	 *
	 * we found that the optimal parameters differ. in both cases, the relevant
	 * parameters were the patch distance threshold and the b_t coefficient that
	 * controls the amount of temporal averaging.
	 *
	 * with the derfhd videos, the distance threshold is lower (i.e. patches
	 * are required to be at a smallest distance to be considered 'similar',
	 * and the temporal averaging is higher.
	 *
	 * the reason for the lowest distance threshold is that the derfhd videos 
	 * are smoother than the cif ones. in fact, the cif ones are not properly
	 * sampled and have a considerable amount of aliasing. this high frequencies
	 * increase the distance between similar patches. 
	 *
	 * i don't know which might be the reason for the increase in the temporal 
	 * averaging factor. perhaps that (a) the optical flow can be better estimated
	 * (b) there are more homogeneous regions. in the case of (b), even if the oflow
	 * is not correct, increasing the temporal averaging at these homogeneous regions
	 * might lead to a global decrease in psnr */
#define DERFHD_PARAMS
#ifdef DERFHD_PARAMS
	if (p->patch_sz      < 0) p->patch_sz      = 8;  // not tuned
	if (p->search_sz     < 0) p->search_sz     = 10; // not tuned
 #ifndef K_SIMILAR_PATCHES
	if (p->dista_th      < 0) p->dista_th      = .5*sigma + 15.0;
 #else
	if (p->num_patches_x < 0) p->num_patches_x = 32;
	if (p->num_patches_t < 0) p->num_patches_t = 32;
	if (p->num_patches_tx < 0) p->num_patches_tx = 1;
 #endif
	if (p->dista_lambda  < 0) p->dista_lambda  = 1.0;
	if (p->beta_x        < 0) p->beta_x        = 3.0;
	if (p->beta_t        < 0) p->beta_t        = 0.05*sigma + 6.0;
#else // DERFCIF_PARAMS
	if (p->patch_sz      < 0) p->patch_sz      = 8;  // not tuned
	if (p->search_sz     < 0) p->search_sz     = 10; // not tuned
 #ifndef K_SIMILAR_PATCHES
	if (p->dista_th      < 0) p->dista_th      = (60. - 38.)*(sigma - 10.) + 38.0;
 #else
	if (p->num_patches_x < 0) p->num_patches_x = 32;
	if (p->num_patches_t < 0) p->num_patches_t = 32;
	if (p->num_patches_tx < 0) p->num_patches_tx = 1;
 #endif
	if (p->dista_lambda  < 0) p->dista_lambda  = 1.0;
	if (p->beta_x        < 0) p->beta_x        = 2.4;
	if (p->beta_t        < 0) p->beta_t        = 4.5;
#endif
}

struct patch_distance
{
	int x;
	int y;
	int t;
	float d; // patch distance
};

#ifdef K_SIMILAR_PATCHES
int patch_distance_cmp(const void * a, const void * b)
{
	struct patch_distance * pda = (struct patch_distance *)a;
	struct patch_distance * pdb = (struct patch_distance *)b;
	return (pda->d > pdb->d) - (pda->d < pdb->d);
}

int float_cmp(const void * a, const void * b)
{
	float fa = *(float *)a;
	float fb = *(float *)b;
	return (fa > fb) - (fa < fb);
}
#endif

// nl-kalman filtering of a frame (with k similar patches)
void nlkalman_filter_frame(float *deno1, float *nisy1, float *deno0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame)
{
	// definitions [[[2

	const int psz = prms.patch_sz;
	const int step = prms.pixelwise ? 1 : psz/2;
	const float sigma2 = sigma * sigma;
#ifndef K_SIMILAR_PATCHES
	const float dista_th2 = prms.dista_th * prms.dista_th;
#endif
	const float beta_x  = prms.beta_x;
	const float beta_t  = prms.beta_t;

	// aggregation weights (not necessary for pixel-wise nlmeans)
	float *aggr1 = prms.pixelwise ? NULL : malloc(w*h*sizeof(float));

	// set output and aggregation weights to 0
	for (int i = 0; i < w*h*ch; ++i) deno1[i] = 0.;
	if (aggr1) for (int i = 0; i < w*h; ++i) aggr1[i] = 0.;

	// compute a window (to reduce blocking artifacts)
//	float *window = window_function("gaussian", psz);
	float *window = window_function("constant", psz);
	float W[psz][psz];
	for (int i = 0; i < psz; ++i)
	for (int j = 0; j < psz; ++j)
		W[i][j] = window[i*psz + j];
	free(window);

	// noisy and clean patches at point p (as VLAs in the stack!)
	float N1[psz][psz][ch]; // noisy patch at position p in frame t
	float D0[psz][psz][ch]; // denoised patch at p in frame t - 1

	// wrap images with nice pointers to vlas
	float (*a1)[w]     = (void *)aggr1;       // aggregation weights at t
	float (*d1)[w][ch] = (void *)deno1;       // denoised frame t (output)
	const float (*d0)[w][ch] = (void *)deno0; // denoised frame t-1
	const float (*n1)[w][ch] = (void *)nisy1; // noisy frame at t
	const float (*b1)[w][ch] = (void *)bsic1; // basic estimate frame at t

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
	float M0 [ch][psz][psz]; // average patch at t-1 for spatial filtering
	float M0V[ch][psz][psz]; // average patch at t-1 for variance computation
	float V0 [ch][psz][psz]; // variance at t-1
	float V01[ch][psz][psz]; // transition variance from t-1 to t
	float M1 [ch][psz][psz]; // average patch at t
	float V1 [ch][psz][psz]; // variance at t

	// loop on image patches [[[2
	#pragma omp parallel for private(N1D0,N1,D0,M0,V0,V01,M1,V1,M0V)
	for (int py = 0; py < h - psz + 1; py += step) // FIXME: boundary pixels
	for (int px = 0; px < w - psz + 1; px += step) // may not be denoised
	{
		//	load target patch [[[3
		bool prev_p = d0;
		for (int hy = 0; hy < psz; ++hy)
		for (int hx = 0; hx < psz; ++hx)
		{
			if (prev_p && isnan(d0[py + hy][px + hx][0])) prev_p = false;
			for (int c  = 0; c  < ch ; ++c )
			{
				D0[hy][hx][c] = prev_p ? d0[py + hy][px + hx][c] : 0.f;
				N1[hy][hx][c] = b1     ? b1[py + hy][px + hx][c]
				                       : n1[py + hy][px + hx][c];

				M0 [c][hy][hx] = 0.;
				M0V[c][hy][hx] = 0.;
				V0 [c][hy][hx] = 0.;
				M1 [c][hy][hx] = 0.;
				V1 [c][hy][hx] = 0.;
				V01[c][hy][hx] = 0.;
			}
		}

		// gather spatio-temporal statistics: loop on search region [[[3
		int np0 = 0; // number of similar patches with a  valid previous patch
		int np1 = 0; // number of similar patches with no valid previous patch
#ifdef K_SIMILAR_PATCHES
		const float dista_sigma2 = 0; // correct noise in distance
		int num_patches = prev_p ? prms.num_patches_t : prms.num_patches_x;
		if (num_patches)
#else
		const float dista_sigma2 = b1 ? 0 : 2*sigma2; // correct noise in distance
		if (dista_th2)
#endif
		{
			const int wsz = prms.search_sz;
			const int wx[2] = {max(px - wsz, 0), min(px + wsz, w - psz) + 1};
			const int wy[2] = {max(py - wsz, 0), min(py + wsz, h - psz) + 1};

			// compute all distances [[[4
//			float dists[ (wy[1]-wy[0]) * (wx[1]-wx[0]) ];
			struct patch_distance pdists[ (wy[1]-wy[0]) * (wx[1]-wx[0]) ];
			for (int qy = wy[0], i = 0; qy < wy[1]; ++qy)
			for (int qx = wx[0]       ; qx < wx[1]; ++qx, ++i)
			{
				// store patch at q

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
					N1D0[c     ][hy][hx] = b1   ? b1[qy + hy][qx + hx][c]
					                            : n1[qy + hy][qx + hx][c];
					N1D0[c + ch][hy][hx] = prev ? d0[qy + hy][qx + hx][c] : 0;
				}

				// compute patch distance
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
							ww += l * (e1 * e1 - dista_sigma2) + (1 - l) * e0 * e0;
						}
					else
					{
						// use only noisy from current frame
						for (int c  = 0; c  < ch ; ++c )
						{
							const float e1 = N1D0[c][hy][hx] - N1[hy][hx][c];
							ww += e1 * e1 - dista_sigma2;
						}
					}

				// normalize distance by number of pixels in patch
				pdists[i].x = qx;
				pdists[i].y = qy;
				pdists[i].d = max(ww / ((float)psz*psz*ch), 0);
			}

#ifdef K_SIMILAR_PATCHES
			// sort distances [[[4
			qsort(pdists, (wx[1] - wx[0])*(wy[1] - wy[0]), sizeof*pdists, patch_distance_cmp);
			num_patches = min(num_patches, (wy[1]-wy[0]) * (wx[1]-wx[0]));

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
//			for (int i = 0; i < (wy[1]-wy[0]) * (wx[1]-wx[0]); ++i)
//				printf("pdists[%d] = %f - %d, %d\n", i, pdists[i].d,
//				                                        pdists[i].x,
//				                                        pdists[i].y);
//			printf("num_patches = %d\n", num_patches);
//			getchar();
#endif // ]]]
#else
			int num_patches = (wy[1]-wy[0]) * (wx[1]-wx[0]);
#endif

			// gather statistics with patches closer than dista_th2 [[[4
			for (int i = 0; i < num_patches; ++i)
			{
#ifndef K_SIMILAR_PATCHES
				// skip rest of loop if distance is above threshold
				if (pdists[i].d > dista_th2) continue;
#endif
				int qx = pdists[i].x;
				int qy = pdists[i].y;

				// store patch at q [[[5

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
					N1D0[c     ][hy][hx] = b1   ? b1[qy + hy][qx + hx][c]
					                            : n1[qy + hy][qx + hx][c];
					N1D0[c + ch][hy][hx] = prev ? d0[qy + hy][qx + hx][c] : 0;
				}

				// update statistics [[[5
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
							const float oldM0V = M0V[c][hy][hx];
							const float delta = p - oldM0V;

							M0V[c][hy][hx] += delta * inp0;
							V0[c][hy][hx] += delta * (p - M0V[c][hy][hx]);

							p -= N1D0[c][hy][hx];
							V01[c][hy][hx] += p*p;

							if (np0 <= prms.num_patches_tx)
								M0[c][hy][hx] += (N1D0[c + ch][hy][hx] - M0[c][hy][hx]) * inp0;
						}
					}

#ifdef DEBUG_OUTPUT_FILTERING
					if (frame == 2)
					{
						printf("frame = %d\n", frame);
						printf("[%d,%d] np0 = %d\n", qx, qy, np0);
						for (int hy = 0; hy < psz; ++hy)
						{
							for (int hx = 0; hx < psz; ++hx)
								printf("%7.2f ", M0[0][hy][hx]);
							printf("\n");
						}
						getchar();
					}
#endif
				}
			}

			// normalize variance
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
		}
		// local version: single point estimate of variances [[[4
		//                the mean M1 is assumed to be 0
		else // dista_th2 == 0
		{

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

					p -= N1D0[c][hy][hx];
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
			N1D0[c     ][hy][hx] = b1     ? n1[py + hy][px + hx][c]
			                              : N1[hy][hx][c];
			N1D0[c + ch][hy][hx] = prev_p ? D0[hy][hx][c] : 0; //XXX
		}

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
		if (frame == 2)
		{
			printf("Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[0][hy][hx]);
				printf("\n");
			}
		}

		if (frame == 2)
		{
			printf("Previous patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[1][hy][hx]);
				printf("\n");
			}
		}
#endif // ]]]

		// compute dct (computed in place in N1D0)
		dct_threads_forward((float *)N1D0, dcts);

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
		if (frame == 2)
		{
			printf("DCT previous Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[1][hy][hx]);
				printf("\n");
			}
		}
		if (frame == 2)
		{
			printf("DCT Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[0][hy][hx]);
				printf("\n");
			}

			printf("DCT previous Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[1][hy][hx]);
				printf("\n");
			}

			printf("M1 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", M1[0][hy][hx]);
				printf("\n");
			}

			printf("V1 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", V1[0][hy][hx]);
				printf("\n");
			}

			printf("M0V at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", M0V[0][hy][hx]);
				printf("\n");
			}

			printf("V0 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", V0[0][hy][hx]);
				printf("\n");
			}

			printf("V01 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", V01[0][hy][hx]);
				printf("\n");
			}
		}
#endif // ]]]

		float vp = 0;
		if (np0 > 0) // enough patches with a valid previous patch
		{
			// "kalman"-like spatio-temporal denoising

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
		if (frame == 2)
			{
				printf("beta_t = %f - sigma2 = %f\n", beta_t, sigma2);
				printf("Thresholded variances and filters at %d, %d\n", px, py);
				int c  = 0;
				for (int hy = 0; hy < psz; ++hy)
				{
					for (int hx = 0; hx < psz; ++hx)
					{
						// prediction variance (substract sigma2 from group variance)
						float v = V0[c][hy][hx] + max(0.f, V01[c][hy][hx] - (b1 ? 0 : sigma2));

						// wiener filter
						float a = v / (v + beta_t * sigma2);

						printf("%7.2f - %4.2f -- ", v, a);
					}
					printf("\n");
				}
			}
#endif // ]]]

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				// prediction variance (substract sigma2 from transition variance)
				float v = V0[c][hy][hx] + max(0.f, V01[c][hy][hx] - (b1 ? 0 : sigma2));

				// kalman gain
				float a = v / (v + beta_t * sigma2);
				if (a < 0) printf("a = %f v = %f ", a, v);
				if (a > 1) printf("a = %f v = %f ", a, v);

				// variance of filtered patch
//				vp += (1 - a * a) * v - a * a * sigma2; XXX this seemed wrong
				vp += (1 - a * a) * v + a * a * sigma2;

				// filter
//				N1D0[c][hy][hx] = a*N1D0[c][hy][hx] + (1 - a)*N1D0[c + ch][hy][hx];
				N1D0[c][hy][hx] = a*N1D0[c][hy][hx] + (1 - a)*M0[c][hy][hx];

			}
		}
		else // not enough patches with valid previous patch
		{
			// spatial nl-dct using statistics in M1 V1

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
			if (b1)
			{
				printf("Thresholded variances and filters at %d, %d\n", px, py);
				int c  = 0;
				for (int hy = 0; hy < psz; ++hy)
				{
					for (int hx = 0; hx < psz; ++hx)
					{
						// prediction variance (substract sigma2 from group variance)
						float v = max(0.f, V1[c][hy][hx] - (b1 ? 0 : sigma2) );

						// wiener filter
						float a = v / (v + beta_x * sigma2);

						printf("%7.2f - %4.2f -- ", v, a);
					}
					printf("\n");
				}
			}
#endif // ]]]

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				// prediction variance (substract sigma2 from group variance)
				float v = max(0.f, V1[c][hy][hx] - (b1 ? 0 : sigma2) );

				// wiener filter
				float a = v / (v + beta_x * sigma2);
				if (a < 0) printf("a = %f v = %f ", a, v);
				if (a > 1) printf("a = %f v = %f ", a, v);

				// variance of filtered patch
				vp += a * v; // XXX the following was wrong : vp += a * a * v;

				/* thresholding instead of empirical Wiener filtering
				float a = (hy != 0 || hx != 0) ?
				//	(N1D0[c][hy][hx] * N1D0[c][hy][hx] > 3 * sigma2) : 1;
					(v > 1 * sigma2) : 1;
				float a = (hy != 0 || hx != 0) ?
				vp += a;*/

				// filter
				N1D0[c][hy][hx] = a*N1D0[c][hy][hx] + (1 - a)*M1[c][hy][hx];
			}
		}

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
		if (b1)
		{
			printf("DCT of denoised patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[0][hy][hx]);
				printf("\n");
			}
		}
#endif // ]]]

		// invert dct (output in N1D0)
		dct_threads_inverse((float *)N1D0, dcts);

#ifdef DEBUG_OUTPUT_FILTERING // [[[9
		if (b1)
		{
			printf("Denoised patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", N1D0[0][hy][hx]);
				printf("\n");
			}
			getchar();
		}
#endif // ]]]

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
				#pragma omp atomic
				a1[py + hy][px + hx] += w * W[hy][hx];
				for (int c = 0; c < ch ; ++c )
					#pragma omp atomic
					d1[py + hy][px + hx][c] += w * W[hy][hx] * N1D0[c][hy][hx];
			}
		}
		else
			// pixel-wise denoising: aggregate only the central pixel
			for (int c = 0; c < ch ; ++c )
				#pragma omp atomic
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

// nl-kalman smoothing [[[1

// nl-kalman smoothing of a frame (with k similar patches)
void nlkalman_smooth_frame(float *smoo1, float *filt1, float *smoo0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame)
{
	// definitions [[[2

	const int psz = prms.patch_sz;
	const int step = prms.pixelwise ? 1 : psz/2;
//	const int step = psz;
	const float sigma2 = sigma * sigma;
#ifndef K_SIMILAR_PATCHES
	const float dista_th2 = prms.dista_th * prms.dista_th;
#endif
	const float beta_x  = prms.beta_x;
	const float beta_t  = prms.beta_t;

	// aggregation weights (not necessary for pixel-wise nlmeans)
	float *aggr1 = prms.pixelwise ? NULL : malloc(w*h*sizeof(float));

	// set output and aggregation weights to 0
	for (int i = 0; i < w*h*ch; ++i) smoo1[i] = 0.;
	if (aggr1) for (int i = 0; i < w*h; ++i) aggr1[i] = 0.;

	// compute a window (to reduce blocking artifacts)
//	float *window = window_function("gaussian", psz);
	float *window = window_function("constant", psz);
	float W[psz][psz];
	for (int i = 0; i < psz; ++i)
	for (int j = 0; j < psz; ++j)
		W[i][j] = window[i*psz + j];
	free(window);

	// noisy and clean patches at point p (as VLAs in the stack!)
	float F1[psz][psz][ch]; // noisy patch at position p in frame t
	float S0[psz][psz][ch]; // denoised patch at p in frame t - 1

	// wrap images with nice pointers to vlas
	float (*a1)[w]     = (void *)aggr1;       // aggregation weights at t
	float (*s1)[w][ch] = (void *)smoo1;       // denoised frame t (output)
	const float (*s0)[w][ch] = (void *)smoo0; // denoised frame t-1
	const float (*f1)[w][ch] = (void *)filt1; // noisy frame at t
	const float (*b1)[w][ch] = (void *)bsic1; // basic estimate frame at t

	// initialize dct workspaces (we will compute the dct of two patches)
	float F1S0[2*ch][psz][psz]; // noisy patch at t and clean patch at t-1
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
	#pragma omp parallel for private(F1S0,F1,S0,M0,V0,V01,M1,V1)
	for (int py = 0; py < h - psz + 1; py += step) // FIXME: boundary pixels
	for (int px = 0; px < w - psz + 1; px += step) // may not be denoised
	{
		//	load target patch [[[3
		bool prev_p = s0;
		for (int hy = 0; hy < psz; ++hy)
		for (int hx = 0; hx < psz; ++hx)
		{
			if (prev_p && isnan(s0[py + hy][px + hx][0])) prev_p = false;
			for (int c  = 0; c  < ch ; ++c )
			{
				S0[hy][hx][c] = (prev_p) ? s0[py + hy][px + hx][c] : 0.f;
				F1[hy][hx][c] = (b1) ? b1[py + hy][px + hx][c]
				                     : f1[py + hy][px + hx][c];

				M0 [c][hy][hx] = 0.;
				V0 [c][hy][hx] = 0.;
				M1 [c][hy][hx] = 0.;
				V1 [c][hy][hx] = 0.;
				V01[c][hy][hx] = 0.;
			}
		}

		// gather spatio-temporal statistics: loop on search region [[[3
		int np0 = 0; // number of similar patches with a  valid previous patch
		int np1 = 0; // number of similar patches with no valid previous patch
#ifdef K_SIMILAR_PATCHES
		int num_patches = prev_p ? prms.num_patches_t : prms.num_patches_x;
		if (num_patches)
#else
		if (dista_th2)
#endif
		{
			const int wsz = prms.search_sz;
			const int wx[2] = {max(px - wsz, 0), min(px + wsz, w - psz) + 1};
			const int wy[2] = {max(py - wsz, 0), min(py + wsz, h - psz) + 1};

			// compute all distances [[[4
			float dists[ (wy[1]-wy[0]) * (wx[1]-wx[0]) ];
			for (int qy = wy[0], i = 0; qy < wy[1]; ++qy)
			for (int qx = wx[0]       ; qx < wx[1]; ++qx, ++i)
			{
				// store patch at q

				// check if the previous patch at q is valid
				bool prev_q = s0;
				if (prev_q)
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					if (prev_q && isnan(s0[qy + hy][qx + hx][0]))
						prev_q = false;

				const bool prev = prev_p && prev_q;

				for (int c  = 0; c  < ch ; ++c )
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					F1S0[c     ][hy][hx] = b1   ? b1[qy + hy][qx + hx][c]
					                            : f1[qy + hy][qx + hx][c];
					F1S0[c + ch][hy][hx] = prev ? s0[qy + hy][qx + hx][c] : 0;
				}

				// compute patch distance
				float ww = 0; // patch distance is saved here
				const float l = prms.dista_lambda;
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
					if (prev && l != 1)
						// use noisy and denoised patches from previous frame
						for (int c  = 0; c  < ch ; ++c )
						{
							const float e1 = F1S0[c     ][hy][hx] - F1[hy][hx][c];
							const float e0 = F1S0[c + ch][hy][hx] - S0[hy][hx][c];
							ww += l * e1 * e1 + (1 - l) * e0 * e0;
						}
					else
					{
						// use only noisy from current frame
						for (int c  = 0; c  < ch ; ++c )
						{
							const float e1 = F1S0[c][hy][hx] - F1[hy][hx][c];
							ww += e1 * e1;
						}
					}

				// normalize distance by number of pixels in patch
				dists[i] = max(ww / ((float)psz*psz*ch), 0);
			}

#ifdef K_SIMILAR_PATCHES
			// sort dists and find distance to kth nearest patch [[[4
			float sorted_dists[(wx[1] - wx[0])*(wy[1] - wy[0])];
			for (int i = 0; i < (wx[1] - wx[0])*(wy[1] - wy[0]); ++i)
				sorted_dists[i] = dists[i];
			qsort(sorted_dists, (wx[1] - wx[0])*(wy[1] - wy[0]), sizeof*dists, float_cmp);

			num_patches = min(num_patches, (wy[1]-wy[0]) * (wx[1]-wx[0]));
			const float dista_th2 = sorted_dists[num_patches - 1];
#endif

			// gather statistics with patches closer than dista_th2 [[[4
			for (int qy = wy[0], i = 0; qy < wy[1]; ++qy)
			for (int qx = wx[0]       ; qx < wx[1]; ++qx, ++i)
			if (dists[i] <= dista_th2)
			{
				// store patch at q [[[5

				// check if the previous patch at q is valid
				bool prev_q = s0;
				if (prev_q)
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					if (prev_q && isnan(s0[qy + hy][qx + hx][0]))
						prev_q = false;

				const bool prev = prev_p && prev_q;

				for (int c  = 0; c  < ch ; ++c )
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					F1S0[c     ][hy][hx] = b1   ? b1[qy + hy][qx + hx][c]
					                            : f1[qy + hy][qx + hx][c];
					F1S0[c + ch][hy][hx] = prev ? s0[qy + hy][qx + hx][c] : 0;
				}

				// update statistics [[[5
				{
					np1++;
					np0 += prev ? 1 : 0;

					// compute dct (output in F1S0)
					dct_threads_forward((float *)F1S0, dcts);

					// compute means and variances.
					// to compute the variances in a single pass over the search
					// region we use Welford's method.
					const float inp0 = prev ? 1./(float)np0 : 0;
					const float inp1 = 1./(float)np1;
					for (int c  = 0; c  < ch ; ++c )
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					{
						const float p = F1S0[c][hy][hx];
						const float oldM1 = M1[c][hy][hx];
						const float delta = p - oldM1;

						M1[c][hy][hx] += delta * inp1;
						V1[c][hy][hx] += delta * (p - M1[c][hy][hx]); 

						if(prev)
						{
							float p = F1S0[c + ch][hy][hx];
							const float oldM0 = M0[c][hy][hx];
							const float delta = p - oldM0;

							M0[c][hy][hx] += delta * inp0;
							V0[c][hy][hx] += delta * (p - M0[c][hy][hx]);

							p -= F1S0[c][hy][hx];
							V01[c][hy][hx] += p*p;
						}
					}
				}
			}

			// normalize variance
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
		}
		else // dista_th2 == 0
		{
			// local version: single point estimate of variances [[[4
			//                the mean M1 is assumed to be 0

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				F1S0[c     ][hy][hx] =          F1[hy][hx][c];
				F1S0[c + ch][hy][hx] = prev_p ? S0[hy][hx][c] : 0;
			}

			// compute dct (output in F1S0)
			dct_threads_forward((float *)F1S0, dcts);

			// patch statistics (point estimate)
			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				float p = F1S0[c][hy][hx];
				V1[c][hy][hx] = p * p;

				if (prev_p)
				{
					p = F1S0[c + ch][hy][hx];
					V0[c][hy][hx] = p * p;

					p -= F1S0[ch][hy][hx];
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
			F1S0[c     ][hy][hx] = b1     ? f1[py + hy][px + hx][c]
			                              : F1[hy][hx][c];
			F1S0[c + ch][hy][hx] = prev_p ? S0[hy][hx][c] : 0;
		}

#ifdef DEBUG_OUTPUT_SMOOTHING // [[[9
		//if (b1)
		{
			printf("Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", F1S0[0][hy][hx]);
				printf("\n");
			}
		}

		//if (b1)
		{
			printf("Previous patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", F1S0[1][hy][hx]);
				printf("\n");
			}
		}
#endif // ]]]

		// compute dct (computed in place in F1S0)
		dct_threads_forward((float *)F1S0, dcts);

#ifdef DEBUG_OUTPUT_SMOOTHING // [[[9
		//if (b1)
		{
			printf("DCT Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", F1S0[0][hy][hx]);
				printf("\n");
			}

			printf("DCT previous Patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", F1S0[1][hy][hx]);
				printf("\n");
			}

			printf("M1 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", M1[0][hy][hx]);
				printf("\n");
			}

			printf("V1 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", V1[0][hy][hx]);
				printf("\n");
			}

			printf("M0 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", M0[0][hy][hx]);
				printf("\n");
			}

			printf("V0 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", V0[0][hy][hx]);
				printf("\n");
			}

			printf("V01 at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", V01[0][hy][hx]);
				printf("\n");
			}
		}
#endif // ]]]

		float vp = 0;
		if (np0 > 0) // enough patches with a valid previous patch
		{
			// kalman-like temporal smoothing

#ifdef DEBUG_OUTPUT_SMOOTHING // [[[9
			//if (b1)
			{
				printf("beta_t = %f - sigma2 = %f\n", beta_t, sigma2);
				printf("Filters at %d, %d\n", px, py);
				int c  = 0;
				for (int hy = 0; hy < psz; ++hy)
				{
					for (int hx = 0; hx < psz; ++hx)
					{
						// prediction variance (substract sigma2 from group variance)
						float a = V1[c][hy][hx] / (V1[c][hy][hx] + V01[c][hy][hx]);
						printf("%4.2f ", a);
					}
					printf("\n");
				}
			}
#endif // ]]]

			for (int c  = 0; c  < ch ; ++c )
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				// kalman smoothing gain
				float a = V1[c][hy][hx] / (V1[c][hy][hx] + V01[c][hy][hx]);

				// variance of filtered patch
				vp += (1 - a * a) * V1[c][hy][hx]
				    + a * a * max(V0[c][hy][hx] - V01[c][hy][hx], 0.f);

				// filter
				F1S0[c][hy][hx] = (1 - a)*F1S0[c][hy][hx] + a*F1S0[c + ch][hy][hx];
			}
		}
		else // not enough patches with valid previous patch
		{
			// leave filtered version, with small aggregation weights
			vp = 1e-4;
		}

#ifdef DEBUG_OUTPUT_SMOOTHING // [[[9
		//if (b1)
		{
			printf("DCT of denoised patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", F1S0[0][hy][hx]);
				printf("\n");
			}
		}
#endif // ]]]

		// invert dct (output in F1S0)
		dct_threads_inverse((float *)F1S0, dcts);

#ifdef DEBUG_OUTPUT_SMOOTHING // [[[9
		//if (b1)
		{
			printf("Denoised patch at %d, %d\n", px, py);
			for (int hy = 0; hy < psz; ++hy)
			{
				for (int hx = 0; hx < psz; ++hx)
					printf("%7.2f ", F1S0[0][hy][hx]);
				printf("\n");
			}
			printf("posterior variance = %f\n", vp);
			getchar();
		}
#endif // ]]]

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
				#pragma omp atomic
				a1[py + hy][px + hx] += w * W[hy][hx];
				for (int c = 0; c < ch ; ++c )
					#pragma omp atomic
					s1[py + hy][px + hx][c] += w * W[hy][hx] * F1S0[c][hy][hx];
			}
		}
		else
			// pixel-wise denoising: aggregate only the central pixel
			for (int c = 0; c < ch ; ++c )
				#pragma omp atomic
				s1[py + psz/2][px + psz/2][c] += F1S0[c][psz/2][psz/2];

		// ]]]3
	}

	// normalize output [[[2
	if (aggr1)
	for (int i = 0, j = 0; i < w*h; ++i)
	for (int c = 0; c < ch ; ++c, ++j)
		smoo1[j] /= aggr1[i];

	// free allocated mem and quit
	dct_threads_destroy(dcts);
	if (aggr1) free(aggr1);

	return; // ]]]2
}

// main funcion [[[1

// 'usage' message in the command line
static const char *const usages[] = {
	"nlkalman-bwd [options] [[--] args]",
	"nlkalman-bwd [options]",
	NULL,
};

int main(int argc, const char *argv[])
{
	// parse command line [[[2

	// command line parameters and their defaults
	const char *nisy_path = NULL;
	const char *deno_path = NULL;
	const char *bsic_path = NULL;
	const char *bflo_path = NULL;
	const char *bocc_path = NULL;
	const char *fflo_path = NULL;
	const char *focc_path = NULL;
	int fframe = 0, lframe = -1;
	float sigma = 0.f;
	bool verbose = false;
	struct nlkalman_params prms;
	prms.patch_sz     = -1; // -1 means automatic value
	prms.search_sz    = -1;
#ifndef K_SIMILAR_PATCHES
	prms.dista_th     = -1.;
#else
	prms.num_patches_x = -1.;
	prms.num_patches_t = -1.;
	prms.num_patches_tx = -1.;
#endif
	prms.beta_x       = -1.;
	prms.beta_t       = -1.;
	prms.dista_lambda = -1.;
	prms.pixelwise = false;

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Algorithm options"),
		OPT_STRING ('i', "nisy"  , &nisy_path, "noisy input path (printf format)"),
		OPT_STRING ('o', "bflow" , &bflo_path, "bwd flow path (printf format)"),
		OPT_STRING ('k', "boccl" , &bocc_path, "bwd flow occlusions mask (printf format)"),
		OPT_STRING ( 0 , "fflow" , &bflo_path, "fwd flow path (printf format)"),
		OPT_STRING ( 0 , "foccl" , &bocc_path, "fwd flow occlusions mask (printf format)"),
		OPT_STRING ('d', "deno"  , &deno_path, "denoised output path (printf format)"),
		OPT_STRING ( 0 , "bsic"  , &bsic_path, "basic estimate output path (printf format)"),
		OPT_INTEGER('f', "first" , &fframe, "first frame"),
		OPT_INTEGER('l', "last"  , &lframe , "last frame"),
		OPT_FLOAT  ('s', "sigma" , &sigma, "noise standard dev"),
		OPT_INTEGER('p', "patch" , &prms.patch_sz, "patch size"),
		OPT_INTEGER('w', "search", &prms.search_sz, "search region radius"),
#ifndef K_SIMILAR_PATCHES
		OPT_FLOAT  ( 0 , "dth"   , &prms.dista_th, "patch distance threshold"),
#else
		OPT_INTEGER('m', "npatches_x", &prms.num_patches_x, "number of similar patches spatial"),
		OPT_INTEGER('n', "npatches_t", &prms.num_patches_t, "number of similar patches kalman"),
		OPT_INTEGER('n', "npatches_tx", &prms.num_patches_tx, "number of similar patches kalman spatial average"),
#endif
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
	nlkalman_default_params(&prms, sigma);

	// print parameters
	if (verbose)
	{
		printf("parameters:\n");
		printf("\tnoise  %f\n", sigma);
		printf("\t%s-wise mode\n", prms.pixelwise ? "pixel" : "patch");
		printf("\tpatch      %d\n", prms.patch_sz);
		printf("\tsearch     %d\n", prms.search_sz);
#ifndef K_SIMILAR_PATCHES
		printf("\tdth        %g\n", prms.dista_th);
#else
		printf("\tnpatches_x %d\n", prms.num_patches_x);
		printf("\tnpatches_t %d\n", prms.num_patches_t);
		printf("\tnpatches_tx %d\n", prms.num_patches_tx);
#endif
		printf("\tlambda     %g\n", prms.dista_lambda);
		printf("\tbeta_x     %g\n", prms.beta_x);
		printf("\tbeta_t     %g\n", prms.beta_t);
		printf("\n");
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

	// load backward optical flow [[[3
	float * bflo = NULL;
	if (bflo_path)
	{
		if (verbose) printf("loading bwd flow %s\n", bflo_path);
		int w1, h1, c1;
		bflo = vio_read_video_float_vec(bflo_path, fframe, lframe, &w1, &h1, &c1);

		if (!bflo)
		{
			if (nisy) free(nisy);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 2)
		{
			fprintf(stderr, "Video and bwd optical flow size missmatch\n");
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			return EXIT_FAILURE;
		}
	}

	// load forward optical flow [[[3
	float * fflo = NULL;
	if (fflo_path)
	{
		if (verbose) printf("loading bwd flow %s\n", fflo_path);
		int w1, h1, c1;
		fflo = vio_read_video_float_vec(fflo_path, fframe, lframe, &w1, &h1, &c1);

		if (!fflo)
		{
			if (nisy) free(nisy);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 2)
		{
			fprintf(stderr, "Video and bwd optical flow size missmatch\n");
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (fflo) free(fflo);
			return EXIT_FAILURE;
		}
	}

	// load backward occlusion masks [[[3
	float * bocc = NULL;
	if (bflo_path && bocc_path)
	{
		if (verbose) printf("loading bwd occl. mask %s\n", bocc_path);
		int w1, h1, c1;
		bocc = vio_read_video_float_vec(bocc_path, fframe, lframe, &w1, &h1, &c1);

		if (!bocc)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (fflo) free(fflo);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 1)
		{
			fprintf(stderr, "Video and bwd occl. masks size missmatch\n");
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (fflo) free(fflo);
			if (bocc) free(bocc);
			return EXIT_FAILURE;
		}
	}

	// load forward occlusion masks [[[3
	float * focc = NULL;
	if (fflo_path && focc_path)
	{
		if (verbose) printf("loading fwd occl. mask %s\n", focc_path);
		int w1, h1, c1;
		focc = vio_read_video_float_vec(focc_path, fframe, lframe, &w1, &h1, &c1);

		if (!focc)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (fflo) free(fflo);
			if (bocc) free(bocc);
			return EXIT_FAILURE;
		}

		if (w*h != w1*h1 || c1 != 1)
		{
			fprintf(stderr, "Video and fwd occl. masks size missmatch\n");
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (fflo) free(fflo);
			if (bocc) free(bocc);
			if (focc) free(focc);
			return EXIT_FAILURE;
		}
	}

	// run denoiser [[[2
	char frame_name[512];
	const int whc = w*h*c, wh2 = w*h*2;
	float * deno = nisy;
	float * warp0 = malloc(whc*sizeof(float));
	float * bsic1 = malloc(whc*sizeof(float));
	float * deno1 = malloc(whc*sizeof(float));
	for (int f = fframe; f <= lframe; ++f)
	{
		if (verbose) printf("processing frame %d\n", f);

		// warp previous denoised frame [[[3
		if (f > fframe)
		{
			float * deno0 = deno + (f - 1 - fframe)*whc;
			if (bflo)
			{
				float * flow0 = bflo + (f - fframe)*wh2;
				float * occl1 = bocc ? bocc + (f - fframe)*w*h : NULL;
				warp_bicubic(warp0, deno0, flow0, occl1, w, h, c);
			}
			else
				// copy without warping
				memcpy(warp0, deno0, whc*sizeof(float));
		}

		// filtering 1st step [[[3
		float *nisy1 = nisy + (f - fframe)*whc;
		float *deno0 = (f > fframe) ? warp0 : NULL;
		nlkalman_filter_frame(bsic1, nisy1, deno0, NULL, w, h, c, sigma, prms, f);

		// filtering 2nd step [[[3
		bool second_step = true;
		if (second_step && f > fframe)
		{
			// save output
			sprintf(frame_name, "/tmp/bsic-%03d.png", f);
			iio_save_image_float_vec(frame_name, bsic1, w, h, c);

			// second step
			struct nlkalman_params prms2;
			prms2.patch_sz      = prms.patch_sz; // -1 means automatic value
			prms2.search_sz     = prms.search_sz;
#ifdef K_SIMILAR_PATCHES
			prms2.num_patches_x = prms.num_patches_x;
			prms2.num_patches_t = prms.num_patches_t;
			prms2.num_patches_tx = prms.num_patches_tx;
#else
			prms2.dista_th      = prms.dista_th;
#endif
			prms2.beta_x        = prms.beta_x;
			prms2.beta_t        = prms.beta_t;
			prms2.dista_lambda  = prms.dista_lambda;
			prms2.pixelwise     = false;

			nlkalman_filter_frame(deno1, nisy1, deno0, bsic1, w, h, c, sigma, prms2, f);
			memcpy(nisy1, deno1, whc*sizeof(float));

		}
		else
			memcpy(nisy1, bsic1, whc*sizeof(float));

		// save output
		sprintf(frame_name, deno_path, f);
		iio_save_image_float_vec(frame_name, nisy1, w, h, c);

		// smoothing [[[3
		bool smoothing = true;
		if (smoothing && f > fframe)
		{
			// point to previous frame
			float *filt1 = deno + (f - 1 - fframe)*whc;

			// warp to previous frame [[[3
			float * smoo0 = deno + (f - fframe)*whc;
			{
				if (fflo)
				{
					float * flow0 = fflo + (f - 1 - fframe)*wh2;
					float * occl0 = focc ? focc + (f - 1 - fframe)*w*h : NULL;
					warp_bicubic(warp0, smoo0, flow0, occl0, w, h, c);
				}
				else
					// copy without warping
					memcpy(warp0, smoo0, whc*sizeof(float));

				smoo0 = warp0;
			}

			// smoothing parameters
			struct nlkalman_params prms2;
			prms2.patch_sz     = prms.patch_sz; // -1 means automatic value
			prms2.search_sz    = prms.search_sz;
#ifdef K_SIMILAR_PATCHES
			prms2.num_patches_x = prms.num_patches_x;
			prms2.num_patches_t = prms.num_patches_t;
			prms2.num_patches_tx = prms.num_patches_tx;
#else
			prms2.dista_th     = prms.dista_th;
#endif
			prms2.beta_x       = prms.beta_x;
			prms2.beta_t       = prms.beta_t;
			prms2.dista_lambda = prms.dista_lambda;
			prms2.pixelwise = false;

			nlkalman_smooth_frame(deno1, filt1, smoo0, NULL, w, h, c, sigma, prms2, f);
			memcpy(filt1, deno1, whc*sizeof(float));

			// save output
			sprintf(frame_name, "/tmp/smoo-%03d.png", f-1);
			iio_save_image_float_vec(frame_name, deno1, w, h, c);
		}
	}

	// save output [[[2
//	vio_save_video_float_vec(deno_path, deno, fframe, lframe, w, h, c);

	if (deno1) free(deno1);
	if (bsic1) free(bsic1);
	if (warp0) free(warp0);
	if (nisy) free(nisy);
	if (bflo) free(bflo);
	if (bocc) free(bocc);
	if (fflo) free(fflo);
	if (focc) free(focc);

	return EXIT_SUCCESS; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
