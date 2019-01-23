#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

#include <stdio.h>     // getchar() for debugging

// some macros and data types [[[1

// comment to decouple the 1st filtering stage from the 2nd
#define DECOUPLE_FILTER2

// comment for patch distance using previous frame
//#define LAMBDA_DISTANCE

// comment for uniform aggregation
#define WEIGHTED_AGGREGATION

// comment for distance threshold
#define K_SIMILAR_PATCHES

// for debugging
//#define DEBUG_OUTPUT_FILTERING
//#define DEBUG_OUTPUT_SMOOTHING

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
	int search_sz_x;     // search window radius for spatial filtering
	int search_sz_t;     // search window radius for temporal filtering
#ifdef K_SIMILAR_PATCHES
	int npatches_x;      // number of similar patches spatial filtering
	int npatches_t;      // number of similar patches temporal filtering
	int npatches_tagg;   // number of similar patches statial average in temporal filtering
#else
	float dista_th;      // patch distance threshold
#endif
	float dista_lambda;  // weight of current frame in patch distance
	float beta_x;        // noise multiplier in spatial filtering
	float beta_t;        // noise multiplier in kalman filtering
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
	if (p->search_sz_x   < 0) p->search_sz_x   = 10; // not tuned
	if (p->search_sz_t   < 0) p->search_sz_t   = 10; // not tuned
 #ifndef K_SIMILAR_PATCHES
	if (p->dista_th      < 0) p->dista_th      = .5*sigma + 15.0;
 #else
	if (p->npatches_x    < 0) p->npatches_x    = 32;
	if (p->npatches_t    < 0) p->npatches_t    = 32;
	if (p->npatches_tagg < 0) p->npatches_tagg = 1;
 #endif
	if (p->dista_lambda  < 0) p->dista_lambda  = 1.0;
	if (p->beta_x        < 0) p->beta_x        = 3.0;
	if (p->beta_t        < 0) p->beta_t        = 0.05*sigma + 6.0;
#else // DERFCIF_PARAMS
	if (p->patch_sz      < 0) p->patch_sz      = 8;  // not tuned
	if (p->search_sz_x   < 0) p->search_sz_x   = 10; // not tuned
	if (p->search_sz_t   < 0) p->search_sz_t   = 10; // not tuned
 #ifndef K_SIMILAR_PATCHES
	if (p->dista_th      < 0) p->dista_th      = (60. - 38.)*(sigma - 10.) + 38.0;
 #else
	if (p->npatches_x    < 0) p->npatches_x    = 32;
	if (p->npatches_t    < 0) p->npatches_t    = 32;
	if (p->npatches_tagg < 0) p->npatches_tagg = 1;
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

//#define DCT_IMAGE
#ifndef DCT_IMAGE
// nl-kalman filtering of a frame (with k similar patches)
void nlkalman_filter_frame(float *deno1, float *nisy1, float *deno0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame)
{
	// definitions [[[2

	const int psz = prms.patch_sz;
	const int step = psz/2;
	const float sigma2 = sigma * sigma;
#ifndef K_SIMILAR_PATCHES
	const float dista_th2 = prms.dista_th * prms.dista_th;
#endif
	const float beta_x  = prms.beta_x;
	const float beta_t  = prms.beta_t;

	// aggregation weights
	float *aggr1 = malloc(w*h*sizeof(float));
	int   *mask1 = malloc(w*h*sizeof(int));

	// set output and aggregation weights to 0
	for (int i = 0; i < w*h*ch; ++i) deno1[i] = 0.;
	for (int i = 0; i < w*h; ++i) aggr1[i] = 0., mask1[i] = 0;

	// compute a window (to reduce blocking artifacts)
	float *window = window_function("gaussian", psz);
//	float *window = window_function("constant", psz);
	float W[psz][psz];
	for (int i = 0; i < psz; ++i)
	for (int j = 0; j < psz; ++j)
		W[i][j] = window[i*psz + j];
	free(window);

	// noisy and clean patches at point p (as VLAs in the stack!)
	float N1[psz][psz][ch]; // noisy patch at position p in frame t
	float D0[psz][psz][ch]; // denoised patch at p in frame t - 1

	// wrap images with nice pointers to vlas
	int   (*m1)[w]     = (void *)mask1;       // mask of processed patches at t
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

	// dct transform for the whole group
	struct dct_threads dcts_pg[1];
	dct_threads_init(psz, psz, 1, prms.npatches_tagg*ch, nthreads, dcts_pg);

	// statistics
	float M0 [ch][psz][psz]; // average patch at t-1 for spatial filtering
	float M0V[ch][psz][psz]; // average patch at t-1 for variance computation
	float V0 [ch][psz][psz]; // variance at t-1
	float V01[ch][psz][psz]; // transition variance from t-1 to t
	float M1 [ch][psz][psz]; // average patch at t
	float V1 [ch][psz][psz]; // variance at t

	// loop on image patches [[[2
	#pragma omp parallel for private(N1D0,N1,D0,M0,V0,V01,M1,V1,M0V)
	for (int py = 0; py < h - psz + 1; py += step) // FIXME: bottom image border
	{
		// aggregation patch group
		int nagg = prms.npatches_tagg;
		float * patch_group = (float *)malloc(nagg*ch*psz*psz*sizeof*patch_group);
		float (*PG)[ch][psz][psz] = (void *)patch_group;
		struct patch_distance patch_group_coords[ nagg ];

		for (int px = 0; px < w - psz + 1; px += step) // FIXME: right image border
		{
			int mask_p;
			#pragma omp atomic read
			mask_p = m1[py][px];
			if (mask_p) continue;

			int nagg = prms.npatches_tagg;

			// load target patch [[[3
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
			int num_patches = prev_p ? prms.npatches_t : prms.npatches_x;
			if (num_patches > 1)
#else
			const float dista_sigma2 = b1 ? 0 : 2*sigma2; // correct noise in distance
			if (dista_th2)
#endif
			{
				const int wsz = prev_p ? prms.search_sz_t : prms.search_sz_x ;
				const int wx[2] = {max(px - wsz, 0), min(px + wsz, w - psz) + 1};
				const int wy[2] = {max(py - wsz, 0), min(py + wsz, h - psz) + 1};

				// compute all distances [[[4
				const float l = prms.dista_lambda;
				struct patch_distance pdists[ (wy[1]-wy[0]) * (wx[1]-wx[0]) ];
				for (int qy = wy[0], i = 0; qy < wy[1]; ++qy)
				for (int qx = wx[0]       ; qx < wx[1]; ++qx, ++i)
				{
#ifdef LAMBDA_DISTANCE // slow patch distance [[[5
					// check if the previous patch at q is valid
					bool prev = false;
					if (l != 1)
					{
						bool prev_q = d0;
						if (prev_q)
							for (int hy = 0; hy < psz; ++hy)
							for (int hx = 0; hx < psz; ++hx)
							if (prev_q && isnan(d0[qy + hy][qx + hx][0]))
								prev_q = false;

						prev = prev_p && prev_q;
					}

					// compute patch distance
					float ww = 0; // patch distance is saved here
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
						if (prev)
							// use noisy and denoised patches from previous frame
							for (int c  = 0; c  < ch ; ++c )
							{
								const float e1 = b1 ? b1[qy + hy][qx + hx][c] - N1[hy][hx][c]
								                    : n1[qy + hy][qx + hx][c] - N1[hy][hx][c];
								const float e0 = d0[qy + hy][qx + hx][c] - D0[hy][hx][c];
								ww += l * (e1 * e1 - dista_sigma2) + (1 - l) * e0 * e0;
							}
						else
						{
							// use only noisy from current frame
							for (int c  = 0; c  < ch ; ++c )
							{
								const float e1 = b1 ? b1[qy + hy][qx + hx][c] - N1[hy][hx][c]
								                    : n1[qy + hy][qx + hx][c] - N1[hy][hx][c];
								ww += e1 * e1 - dista_sigma2;
							}
						} // 5]]]
#else // faster version of the distance (when lambda = 1)
					// compute patch distance
					float ww = 0; // patch distance is saved here
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					for (int c  = 0; c  < ch ; ++c )
					{
						const float e1 = b1 ? b1[qy + hy][qx + hx][c] - N1[hy][hx][c]
						                    : n1[qy + hy][qx + hx][c] - N1[hy][hx][c];
						ww += e1 * e1 - dista_sigma2;
					}
#endif

					// normalize distance by number of pixels in patch
					pdists[i].x = qx;
					pdists[i].y = qy;
					pdists[i].d = max(ww / ((float)psz*psz*ch), 0);
				}

#ifdef K_SIMILAR_PATCHES
				// sort distances [[[4
				qsort(pdists, (wx[1] - wx[0])*(wy[1] - wy[0]), sizeof*pdists, patch_distance_cmp);
				num_patches = min(num_patches, (wy[1]-wy[0]) * (wx[1]-wx[0]));
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

					// compute dct (output in N1D0)
					dct_threads_forward((float *)N1D0, dcts);

					// update statistics [[[5
					{
						np1++;
						np0 += prev ? 1 : 0;

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

								if (np0 <= prms.npatches_tagg)
								{
									patch_group_coords[np0-1].x = qx;
									patch_group_coords[np0-1].y = qy;
									M0[c][hy][hx] += (N1D0[c + ch][hy][hx] - M0[c][hy][hx]) * inp0;
									PG[np0-1][c][hy][hx] = b1 ? n1[qy + hy][qx + hx][c]
									                          : N1D0[c][hy][hx];
								}
							}
							else if (np1 <= prms.npatches_tagg)
							{
								patch_group_coords[np1-1].x = qx;
								patch_group_coords[np1-1].y = qy;
								PG[np1-1][c][hy][hx] = b1 ? n1[qy + hy][qx + hx][c] : N1D0[c][hy][hx];
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
					PG[0][c][hy][hx] = b1 ? n1[py + hy][px + hx][c] : p;
					V1[c][hy][hx] = p * p;

					if (prev_p)
					{
						p = N1D0[c + ch][hy][hx];
						V0[c][hy][hx] = p * p;

						M0[c][hy][hx] = p;

						p -= N1D0[c][hy][hx];
						V01[c][hy][hx] = p * p;
					}
				}//]]]4
			}

			// filter patch group [[[3

			if (b1) dct_threads_forward((float *)PG, dcts_pg);

			float vp = 0;
//			nagg = min(np0 ? np0 : (d0 ? 1 : np1), prms.npatches_tagg);
			nagg = min(np0 ? np0 : np1, prms.npatches_tagg);
			for (int n = 0; n < nagg; ++n)
			if (np0 > 0) // enough patches with a valid previous patch [[[4
			{
				// "kalman"-like spatio-temporal denoising

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
					vp += (1 - a * a) * v + a * a * sigma2;

					// filter
					PG[n][c][hy][hx] = a*PG[n][c][hy][hx] + (1 - a)*M0[c][hy][hx];

				}
			}
			else // not enough patches with valid previous patch [[[4
			{
				// spatial nl-dct using statistics in M1 V1
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

					// filter
					PG[n][c][hy][hx] = a*PG[n][c][hy][hx] + (1 - a)*M1[c][hy][hx];
				}
			}

			dct_threads_inverse((float *)PG, dcts_pg);

			// aggregate denoised group on output image [[[3
#ifdef WEIGHTED_AGGREGATION
//			const float w = (d0 && !np0) ? 1e-6 : 1.f/vp;
			const float w = 1.f/max(vp, 1e-6);
#else
//			const float w = (d0 && !np0) ? 1e-6 : 1.f;
			const float w = 1.f;
#endif
			for (int n = 0; n < nagg; ++n)
			{
				int qx = patch_group_coords[n].x;
				int qy = patch_group_coords[n].y;
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					#pragma omp atomic
					a1[qy + hy][qx + hx] += w * W[hy][hx];
					for (int c = 0; c < ch ; ++c )
						#pragma omp atomic
						d1[qy + hy][qx + hx][c] += w * W[hy][hx] * PG[n][c][hy][hx];
				}

				#pragma omp atomic
				m1[qy][qx] += (d0 && !np0) ? 0 : 1;
			}

			// ]]]3
		}
		if (patch_group) free(patch_group);
	}

	// normalize output [[[2
	for (int i = 0, j = 0; i < w*h; ++i)
	for (int c = 0; c < ch ; ++c, ++j)
		deno1[j] /= aggr1[i];

	// free allocated mem and quit
	dct_threads_destroy(dcts);
	dct_threads_destroy(dcts_pg);
	if (aggr1) free(aggr1);
	if (mask1) free(mask1);

	return; // ]]]2
}
#else
// nl-kalman filtering of a frame (with k similar patches)
void nlkalman_filter_frame(float *deno1, float *nisy1, float *deno0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame)
{
	// definitions [[[2

	const int psz = prms.patch_sz;
	const int step = psz/2;
	const float sigma2 = sigma * sigma;
#ifndef K_SIMILAR_PATCHES
	const float dista_th2 = prms.dista_th * prms.dista_th;
#endif
	const float beta_x  = prms.beta_x;
	const float beta_t  = prms.beta_t;

	// aggregation weights
	float *aggr1 = malloc(w*h*sizeof(float));
	int   *mask1 = malloc(w*h*sizeof(int));

	// set output and aggregation weights to 0
	for (int i = 0; i < w*h*ch; ++i) deno1[i] = 0.;
	for (int i = 0; i < w*h; ++i) aggr1[i] = 0., mask1[i] = 0;

	// compute a window (to reduce blocking artifacts)
	float *window = window_function("gaussian", psz);
//	float *window = window_function("constant", psz);
	float W[psz][psz];
	for (int i = 0; i < psz; ++i)
	for (int j = 0; j < psz; ++j)
		W[i][j] = window[i*psz + j];
	free(window);

	// noisy and clean patches at point p (as VLAs in the stack!)
	float N1[psz][psz][ch]; // noisy patch at position p in frame t
	float D0[psz][psz][ch]; // denoised patch at p in frame t - 1

	// wrap images with nice pointers to vlas
	int   (*m1)[w]     = (void *)mask1;       // mask of processed patches at t
	float (*a1)[w]     = (void *)aggr1;       // aggregation weights at t
	float (*d1)[w][ch] = (void *)deno1;       // denoised frame t (output)
	const float (*d0)[w][ch] = (void *)deno0; // denoised frame t-1
	const float (*n1)[w][ch] = (void *)nisy1; // noisy frame at t
	const float (*b1)[w][ch] = (void *)bsic1; // basic estimate frame at t

	// initialize dct workspaces (we will compute the dct of two patches)
#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
#else
	const int nthreads = 1;
#endif

	// dct transform for the patch group
	struct dct_threads dcts_pg[1];
	dct_threads_init(psz, psz, 1, prms.npatches_tagg*ch, nthreads, dcts_pg);

	// statistics
	float M0 [ch][psz][psz]; // average patch at t-1 for spatial filtering
	float M0V[ch][psz][psz]; // average patch at t-1 for variance computation
	float V0 [ch][psz][psz]; // variance at t-1
	float V01[ch][psz][psz]; // transition variance from t-1 to t
	float M1 [ch][psz][psz]; // average patch at t
	float V1 [ch][psz][psz]; // variance at t

	// compute dct images [[[2
	const int hp = h - psz + 1;
	const int wp = w - psz + 1;
	float *dct_nisy1 =      (float*)malloc(ch*psz*psz*hp*wp*sizeof(float));
	float *dct_deno0 = d0 ? (float*)malloc(ch*psz*psz*hp*wp*sizeof(float)): NULL;
	float *dct_bsic1 = b1 ? (float*)malloc(ch*psz*psz*hp*wp*sizeof(float)): NULL;
	float (* dctn1)[wp][ch][psz][psz] = (void *)dct_nisy1;
	float (* dctb1)[wp][ch][psz][psz] = (void *)dct_bsic1;
	float (* dctd0)[wp][ch][psz][psz] = (void *)dct_deno0;

	{
		const int num_images = 1 + (d0 ? 1 : 0) + (b1 ? 1 : 0);
		float N1B1D0[num_images*ch][psz][psz]; // buffer for dcts
		struct dct_threads dct_patch[1];
		dct_threads_init(psz, psz, 1, num_images*ch, nthreads, dct_patch);
		#pragma omp parallel for private(N1B1D0)
		for (int py = 0; py < hp; ++py)
		for (int px = 0; px < wp; ++px)
		{
			// load target patch
			bool prev_p = d0;
			for (int hy = 0; hy < psz; ++hy)
			for (int hx = 0; hx < psz; ++hx)
			{
				if (prev_p && isnan(d0[py + hy][px + hx][0])) prev_p = false;
				for (int c  = 0; c  < ch ; ++c)
				{
					int n = 0;   N1B1D0[c + n*ch][hy][hx] = n1[py + hy][px + hx][c];
					if (b1) n++, N1B1D0[c + n*ch][hy][hx] = b1[py + hy][px + hx][c];
					if (d0) n++, N1B1D0[c + n*ch][hy][hx] = prev_p ? d0[py + hy][px + hx][c] : 0.;
				}
			}

			// compute dct transform
			dct_threads_forward((float *)N1B1D0, dct_patch);

			// store in dct images
			int n = 0;   memcpy(dctn1[py][px], N1B1D0[n*ch], ch*psz*psz*sizeof(float));
			if (b1) n++, memcpy(dctb1[py][px], N1B1D0[n*ch], ch*psz*psz*sizeof(float));
			if (d0) n++, memcpy(dctd0[py][px], N1B1D0[n*ch], ch*psz*psz*sizeof(float));
		}
		dct_threads_destroy(dct_patch);
	}

	// loop on image patches [[[2
	#pragma omp parallel for private(N1,D0,M0,V0,V01,M1,V1,M0V)
	for (int py = 0; py < h - psz + 1; py += step) // FIXME: bottom image border
	{
		// aggregation patch group
		int nagg = prms.npatches_tagg;
		float * patch_group = (float *)malloc(nagg*ch*psz*psz*sizeof*patch_group);
		float (*PG)[ch][psz][psz] = (void *)patch_group;
		struct patch_distance patch_group_coords[ nagg ];

		for (int px = 0; px < w - psz + 1; px += step) // FIXME: right image border
		{
			int mask_p;
			#pragma omp atomic read
			mask_p = m1[py][px];
			if (mask_p) continue;

			int nagg = prms.npatches_tagg;

			// load target patch [[[3
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
			int num_patches = prev_p ? prms.npatches_t : prms.npatches_x;
			if (num_patches > 1)
#else
			const float dista_sigma2 = b1 ? 0 : 2*sigma2; // correct noise in distance
			if (dista_th2)
#endif
			{
				const int wsz = prev_p ? prms.search_sz_t : prms.search_sz_x ;
				const int wx[2] = {max(px - wsz, 0), min(px + wsz, w - psz) + 1};
				const int wy[2] = {max(py - wsz, 0), min(py + wsz, h - psz) + 1};

				// compute all distances [[[4
				const float l = prms.dista_lambda;
				struct patch_distance pdists[ (wy[1]-wy[0]) * (wx[1]-wx[0]) ];
				for (int qy = wy[0], i = 0; qy < wy[1]; ++qy)
				for (int qx = wx[0]       ; qx < wx[1]; ++qx, ++i)
				{
#ifdef LAMBDA_DISTANCE // slow patch distance [[[5
					// check if the previous patch at q is valid
					bool prev = false;
					if (l != 1)
					{
						bool prev_q = d0;
						if (prev_q)
							for (int hy = 0; hy < psz; ++hy)
							for (int hx = 0; hx < psz; ++hx)
							if (prev_q && isnan(d0[qy + hy][qx + hx][0]))
								prev_q = false;

						prev = prev_p && prev_q;
					}

					// compute patch distance
					float ww = 0; // patch distance is saved here
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
						if (prev)
							// use noisy and denoised patches from previous frame
							for (int c  = 0; c  < ch ; ++c )
							{
								const float e1 = b1 ? b1[qy + hy][qx + hx][c] - N1[hy][hx][c]
								                    : n1[qy + hy][qx + hx][c] - N1[hy][hx][c];
								const float e0 = d0[qy + hy][qx + hx][c] - D0[hy][hx][c];
								ww += l * (e1 * e1 - dista_sigma2) + (1 - l) * e0 * e0;
							}
						else
						{
							// use only noisy from current frame
							for (int c  = 0; c  < ch ; ++c )
							{
								const float e1 = b1 ? b1[qy + hy][qx + hx][c] - N1[hy][hx][c]
								                    : n1[qy + hy][qx + hx][c] - N1[hy][hx][c];
								ww += e1 * e1 - dista_sigma2;
							}
						} // 5]]]
#else // faster version of the distance (when lambda = 1)
					// compute patch distance
					float ww = 0; // patch distance is saved here
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					for (int c  = 0; c  < ch ; ++c )
					{
						const float e1 = b1 ? b1[qy + hy][qx + hx][c] - N1[hy][hx][c]
						                    : n1[qy + hy][qx + hx][c] - N1[hy][hx][c];
						ww += e1 * e1 - dista_sigma2;
					}
#endif

					// normalize distance by number of pixels in patch
					pdists[i].x = qx;
					pdists[i].y = qy;
					pdists[i].d = max(ww / ((float)psz*psz*ch), 0);
				}

#ifdef K_SIMILAR_PATCHES
				// sort distances [[[4
				qsort(pdists, (wx[1] - wx[0])*(wy[1] - wy[0]), sizeof*pdists, patch_distance_cmp);
				num_patches = min(num_patches, (wy[1]-wy[0]) * (wx[1]-wx[0]));
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

					// update statistics [[[5
					{
						np1++;
						np0 += prev ? 1 : 0;

						float (* N1)[psz][psz] = b1 ? dctb1[qy][qx] : dctn1[qy][qx];
						float (* D0)[psz][psz] =      dctd0[qy][qx];

						// compute means and variances.
						// to compute the variances in a single pass over the search
						// region we use Welford's method.
						const float inp0 = prev ? 1./(float)np0 : 0;
						const float inp1 = 1./(float)np1;
						for (int c  = 0; c  < ch ; ++c )
						for (int hy = 0; hy < psz; ++hy)
						for (int hx = 0; hx < psz; ++hx)
						{
							const float p = N1[c][hy][hx];
							const float oldM1 = M1[c][hy][hx];
							const float delta = p - oldM1;

							M1[c][hy][hx] += delta * inp1;
							V1[c][hy][hx] += delta * (p - M1[c][hy][hx]);

							if(prev)
							{
								float p = D0[c][hy][hx];
								const float oldM0V = M0V[c][hy][hx];
								const float delta = p - oldM0V;

								M0V[c][hy][hx] += delta * inp0;
								V0[c][hy][hx] += delta * (p - M0V[c][hy][hx]);

								p -= N1[c][hy][hx];
								V01[c][hy][hx] += p*p;

								if (np0 <= prms.npatches_tagg)
								{
									patch_group_coords[np0-1].x = qx;
									patch_group_coords[np0-1].y = qy;
									M0[c][hy][hx] += (D0[c][hy][hx] - M0[c][hy][hx]) * inp0;
									PG[np0-1][c][hy][hx] = b1 ? dctn1[qy][qx][c][hy][hx]
									                          : N1[c][hy][hx];
								}
							}
							else if (np1 <= prms.npatches_tagg)
							{
								patch_group_coords[np1-1].x = qx;
								patch_group_coords[np1-1].y = qy;
								PG[np1-1][c][hy][hx] = b1 ? dctn1[qy][qx][c][hy][hx] : N1[c][hy][hx];
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
			// local version: single point estimate of variances [[[4
			//                the mean M1 is assumed to be 0
			else // dista_th2 == 0
			{
				// patch statistics (point estimate)
				for (int c  = 0; c  < ch ; ++c )
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					PG[0][c][hy][hx] = dctn1[py][px][c][hy][hx];
					float p = b1 ? dctb1[py][px][c][hy][hx] : PG[0][c][hy][hx];
					V1[c][hy][hx] = p * p;

					if (prev_p)
					{
						float p0 = dctd0[py][px][c][hy][hx];
						p -= p0;
						V01[c][hy][hx] = p * p;
						V0[c][hy][hx] = p0 * p0;
						M0[c][hy][hx] = p0;
					}
				}//]]]4
			}

			// filter patch group [[[3

			float vp = 0;
			nagg = min(np0 ? np0 : np1, prms.npatches_tagg);
			for (int n = 0; n < nagg; ++n)
			if (np0 > 0) // enough patches with a valid previous patch [[[4
			{
				// "kalman"-like spatio-temporal denoising

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
					vp += (1 - a * a) * v + a * a * sigma2;

					// filter
					PG[n][c][hy][hx] = a*PG[n][c][hy][hx] + (1 - a)*M0[c][hy][hx];

				}
			}
			else // not enough patches with valid previous patch [[[4
			{
				// spatial nl-dct using statistics in M1 V1
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

					// filter
					PG[n][c][hy][hx] = a*PG[n][c][hy][hx] + (1 - a)*M1[c][hy][hx];
				}
			}

			dct_threads_inverse((float *)PG, dcts_pg);

			// aggregate denoised group on output image [[[3
#ifdef WEIGHTED_AGGREGATION
//			const float w = (d0 && !np0) ? 1e-6 : 1.f/vp;
			const float w = 1.f/max(vp, 1e-6);
#else
//			const float w = (d0 && !np0) ? 1e-6 : 1.f;
			const float w = 1.f;
#endif
			for (int n = 0; n < nagg; ++n)
			{
				int qx = patch_group_coords[n].x;
				int qy = patch_group_coords[n].y;
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					#pragma omp atomic
					a1[qy + hy][qx + hx] += w * W[hy][hx];
					for (int c = 0; c < ch ; ++c )
						#pragma omp atomic
						d1[qy + hy][qx + hx][c] += w * W[hy][hx] * PG[n][c][hy][hx];
				}

				#pragma omp atomic
				m1[qy][qx] += (d0 && !np0) ? 0 : 1;
			}

			// ]]]3
		}
		if (patch_group) free(patch_group);
	}

	// normalize output [[[2
	for (int i = 0, j = 0; i < w*h; ++i)
	for (int c = 0; c < ch ; ++c, ++j)
		deno1[j] /= aggr1[i];

	// free allocated mem and quit
	if (aggr1) free(aggr1);
	if (mask1) free(mask1);
	dct_threads_destroy(dcts_pg);
	if (dct_nisy1) free(dct_nisy1);
	if (dct_deno0) free(dct_deno0);
	if (dct_bsic1) free(dct_bsic1);

	return; // ]]]2
}
#endif

// nl-kalman smoothing [[[1

// nl-kalman smoothing of a frame (with k similar patches)
void nlkalman_smooth_frame(float *smoo1, float *filt1, float *smoo0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame)
{
	// definitions [[[2

	const int psz = prms.patch_sz;
	const int step = psz/2;
//	const int step = psz;
	const float sigma2 = sigma * sigma;
#ifndef K_SIMILAR_PATCHES
	const float dista_th2 = prms.dista_th * prms.dista_th;
#endif
	const float beta_x  = prms.beta_x;
	const float beta_t  = prms.beta_t;

	// aggregation weights (not necessary for pixel-wise nlmeans)
	float *aggr1 = malloc(w*h*sizeof(float));
	int   *mask1 = malloc(w*h*sizeof(int));

	// set output and aggregation weights to 0
	for (int i = 0; i < w*h*ch; ++i) smoo1[i] = 0.;
	for (int i = 0; i < w*h; ++i) aggr1[i] = 0., mask1[i] = 0;

	// compute a window (to reduce blocking artifacts)
	float *window = window_function("gaussian", psz);
//	float *window = window_function("constant", psz);
	float W[psz][psz];
	for (int i = 0; i < psz; ++i)
	for (int j = 0; j < psz; ++j)
		W[i][j] = window[i*psz + j];
	free(window);

	// noisy and clean patches at point p (as VLAs in the stack!)
	float F1[psz][psz][ch]; // noisy patch at position p in frame t
	float S0[psz][psz][ch]; // denoised patch at p in frame t - 1

	// wrap images with nice pointers to vlas
	int   (*m1)[w]     = (void *)mask1;       // mask of processed patches at t
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

	// dct transform for the whole group
	struct dct_threads dcts_pg[1];
	dct_threads_init(psz, psz, 1, prms.npatches_tagg*ch, nthreads, dcts_pg);

	// statistics
	float M0 [ch][psz][psz]; // average patch at t-1
	float V0 [ch][psz][psz]; // variance at t-1
	float V01[ch][psz][psz]; // transition variance from t-1 to t
	float M1 [ch][psz][psz]; // average patch at t
	float V1 [ch][psz][psz]; // variance at t

	// loop on image patches [[[2
	#pragma omp parallel for private(F1S0,F1,S0,M0,V0,V01,M1,V1)
	for (int py = 0; py < h - psz + 1; py += step) // FIXME: boundary pixels
	{
		// aggregation patch group
		int nagg = prms.npatches_tagg;
		float * patch_group0 = (float *)malloc(nagg*ch*psz*psz*sizeof(float));
		float * patch_group1 = (float *)malloc(nagg*ch*psz*psz*sizeof(float));
		float (*PG0)[ch][psz][psz] = (void *)patch_group0;
		float (*PG1)[ch][psz][psz] = (void *)patch_group1;
		struct patch_distance patch_group_coords[ nagg ];

		for (int px = 0; px < w - psz + 1; px += step) // may not be denoised
		{
			int mask_p;
			#pragma omp atomic read
			mask_p = m1[py][px];
			if (mask_p) continue;

			int nagg = prms.npatches_tagg;

			// load target patch [[[3
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
			int num_patches = prev_p ? prms.npatches_t : prms.npatches_x;
			if (num_patches > 1)
#else
			if (dista_th2)
#endif
			{
				const int wsz = prms.search_sz_t;
				const int wx[2] = {max(px - wsz, 0), min(px + wsz, w - psz) + 1};
				const int wy[2] = {max(py - wsz, 0), min(py + wsz, h - psz) + 1};

				// compute all distances [[[4
				const float l = prms.dista_lambda;
				struct patch_distance pdists[ (wy[1]-wy[0]) * (wx[1]-wx[0]) ];
				for (int qy = wy[0], i = 0; qy < wy[1]; ++qy)
				for (int qx = wx[0]       ; qx < wx[1]; ++qx, ++i)
				{
#ifdef LAMBDA_DISTANCE // slow patch distance [[[5
					// check if the previous patch at q is valid
					bool prev = false;
					if (l != 1)
					{
						bool prev_q = s0;
						if (prev_q)
							for (int hy = 0; hy < psz; ++hy)
							for (int hx = 0; hx < psz; ++hx)
							if (prev_q && isnan(s0[qy + hy][qx + hx][0]))
								prev_q = false;

						prev = prev_p && prev_q;
					}

					// compute patch distance
					float ww = 0; // patch distance is saved here
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
						if (prev)
							// use noisy and denoised patches from previous frame
							for (int c  = 0; c  < ch ; ++c )
							{
								const float e1 = b1 ? b1[qy + hy][qx + hx][c] - F1[hy][hx][c]
								                    : f1[qy + hy][qx + hx][c] - F1[hy][hx][c];
								const float e0 = s0[qy + hy][qx + hx][c] - S0[hy][hx][c];
								ww += l * e1 * e1 + (1 - l) * e0 * e0;
							}
						else
						{
							// use only noisy from current frame
							for (int c  = 0; c  < ch ; ++c )
							{
								const float e1 = b1 ? b1[qy + hy][qx + hx][c] - F1[hy][hx][c]
								                    : f1[qy + hy][qx + hx][c] - F1[hy][hx][c];
								ww += e1 * e1;
							}
						} // 5]]]
#else // faster version of the distance (when lambda = 1)
					// compute patch distance
					float ww = 0; // patch distance is saved here
					for (int hy = 0; hy < psz; ++hy)
					for (int hx = 0; hx < psz; ++hx)
					for (int c  = 0; c  < ch ; ++c )
					{
						const float e1 = b1 ? b1[qy + hy][qx + hx][c] - F1[hy][hx][c]
						                    : f1[qy + hy][qx + hx][c] - F1[hy][hx][c];
						ww += e1 * e1;
					}
#endif

					// normalize distance by number of pixels in patch
					pdists[i].x = qx;
					pdists[i].y = qy;
					pdists[i].d = max(ww / ((float)psz*psz*ch), 0);
				}

#ifdef K_SIMILAR_PATCHES
				// sort distances [[[4
				qsort(pdists, (wx[1] - wx[0])*(wy[1] - wy[0]), sizeof*pdists, patch_distance_cmp);
				num_patches = min(num_patches, (wy[1]-wy[0]) * (wx[1]-wx[0]));
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

					// compute dct (output in F1S0)
					dct_threads_forward((float *)F1S0, dcts);

					// update statistics [[[5
					{
						np1++;
						np0 += prev ? 1 : 0;

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

								if (np0 <= prms.npatches_tagg)
								{
									patch_group_coords[np0-1].x = qx;
									patch_group_coords[np0-1].y = qy;
									PG0[np0-1][c][hy][hx] = F1S0[c + ch][hy][hx];
									PG1[np0-1][c][hy][hx] = b1 ? f1[qy + hy][qx + hx][c]
									                           : F1S0[c][hy][hx];
								}
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
			// local version: single point estimate of variances [[[4
			//                the mean M1 is assumed to be 0
			else if (prev_p) // dista_th2 == 0
			{
				np0++;

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
					PG1[0][c][hy][hx] = b1 ? f1[py + hy][px + hx][c] : p;
					V1[c][hy][hx] = p * p;

					p = F1S0[c + ch][hy][hx];
					PG0[0][c][hy][hx] = p;
					V0[c][hy][hx] = p * p;

					p -= F1S0[ch][hy][hx];
					V01[c][hy][hx] = p * p;
				}//]]]4
			}

			// filter patch group [[[3

			if (b1) dct_threads_forward((float *)PG1, dcts_pg);

			float vp = 0;
			nagg = min(np0, prms.npatches_tagg);
			const float b = prms.beta_t;
			if (np0 > 0) // enough patches with a valid previous patch
			for (int n = 0; n < nagg; ++n)
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
#endif // 9]]]

				for (int c  = 0; c  < ch ; ++c )
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					// kalman smoothing gain
					float a = V1[c][hy][hx] / (V1[c][hy][hx] + b * V01[c][hy][hx]);

					// variance of filtered patch
					vp += (1 - a * a) * V1[c][hy][hx]
					         + a * a  * max(V0[c][hy][hx] - b * V01[c][hy][hx], 0.f);

					// filter
					PG1[n][c][hy][hx] = (1 - a)*PG1[n][c][hy][hx] + a*PG0[n][c][hy][hx];
				}
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
#endif // 9]]]

			// invert dct (output in F1S0)
			dct_threads_inverse((float *)PG1, dcts_pg);

			if (np0 == 0)
			{
				nagg = 1;
				patch_group_coords[0].x = px;
				patch_group_coords[0].y = py;
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				for (int c  = 0; c  < ch ; ++c )
					PG1[0][c][hy][hx] = f1[py + hy][px + hx][c];
			}

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
#endif // 9]]]

			// aggregate denoised group on output image [[[3
#ifdef WEIGHTED_AGGREGATION
//			const float w = (d0 && !np0) ? 1e-6 : 1.f/vp;
			const float w = 1.f/max(vp, 1e-6);
#else
//			const float w = (d0 && !np0) ? 1e-6 : 1.f;
			const float w = 1.f;
#endif
			for (int n = 0; n < nagg; ++n)
			{
				int qx = patch_group_coords[n].x;
				int qy = patch_group_coords[n].y;
				for (int hy = 0; hy < psz; ++hy)
				for (int hx = 0; hx < psz; ++hx)
				{
					#pragma omp atomic
					a1[qy + hy][qx + hx] += w * W[hy][hx];
					for (int c = 0; c < ch ; ++c )
						#pragma omp atomic
						s1[qy + hy][qx + hx][c] += w * W[hy][hx] * PG1[n][c][hy][hx];
				}

				#pragma omp atomic
				m1[qy][qx] += np0 ? 1 : 0;
			}

			// ]]]3
		}
		if (patch_group0) free(patch_group0);
		if (patch_group1) free(patch_group1);
	}

	// normalize output [[[2
	for (int i = 0, j = 0; i < w*h; ++i)
	for (int c = 0; c < ch ; ++c, ++j)
		smoo1[j] /= aggr1[i];

	// free allocated mem and quit
	dct_threads_destroy(dcts);
	dct_threads_destroy(dcts_pg);
	if (aggr1) free(aggr1);
	if (mask1) free(mask1);

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
	const char *nisy_path = NULL; // input noisy frames path
	const char *bflo_path = NULL; // input bwd flows path
	const char *bocc_path = NULL; // input bwd occlusions path
	const char *fflo_path = NULL; // input fwd flows path
	const char *focc_path = NULL; // input fwd occlusions path
	const char *flt1_path = NULL; // output first filtering path
	const char *flt2_path = NULL; // output second filtering path
	const char *smo1_path = NULL; // output smoothing path
	int fframe = 0, lframe = -1;
	float sigma = 0.f;
	bool verbose = false;

	// first filtering options
	struct nlkalman_params f1_prms;
	f1_prms.patch_sz      = -1; // -1 means automatic value
	f1_prms.search_sz_x   = -1;
	f1_prms.search_sz_t   = -1;
#ifndef K_SIMILAR_PATCHES
	f1_prms.dista_th      = -1.;
#else
	f1_prms.npatches_x    = -1.;
	f1_prms.npatches_t    = -1.;
	f1_prms.npatches_tagg = -1.;
#endif
	f1_prms.beta_x        = -1.;
	f1_prms.beta_t        = -1.;
	f1_prms.dista_lambda  = -1.;

	// second filtering options
	struct nlkalman_params f2_prms;
	f2_prms.patch_sz      = -1; // -1 means automatic value
	f2_prms.search_sz_x   = -1;
	f2_prms.search_sz_t   = -1;
#ifndef K_SIMILAR_PATCHES
	f2_prms.dista_th      = -1.;
#else
	f2_prms.npatches_x    = -1.;
	f2_prms.npatches_t    = -1.;
	f2_prms.npatches_tagg = -1.;
#endif
	f2_prms.beta_x        = -1.;
	f2_prms.beta_t        = -1.;
	f2_prms.dista_lambda  = -1.;

	// smoothing options
	struct nlkalman_params s1_prms;
	s1_prms.patch_sz      =  0; // -1 means automatic value
	s1_prms.search_sz_x   = -1;
	s1_prms.search_sz_t   = -1;
#ifndef K_SIMILAR_PATCHES
	s1_prms.dista_th      = -1.;
#else
	s1_prms.npatches_x    = -1.;
	s1_prms.npatches_t    = -1.;
	s1_prms.npatches_tagg = -1.;
#endif
	s1_prms.beta_x        = -1.;
	s1_prms.beta_t        = -1.;
	s1_prms.dista_lambda  = -1.;

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Data i/o options (all paths in printf format)"),
		OPT_STRING ('i', "nisy"  , &nisy_path, "input noisy frames path"),
		OPT_STRING ('o', "bflow" , &bflo_path, "input bwd flow path"),
		OPT_STRING ('k', "boccl" , &bocc_path, "input bwd occlusion masks path"),
		OPT_STRING ( 0 , "fflow" , &fflo_path, "input fwd flow path"),
		OPT_STRING ( 0 , "foccl" , &focc_path, "input fwd occlusion masks path"),
		OPT_STRING ( 0 , "filt1" , &flt1_path, "output first filtering path"),
		OPT_STRING ( 0 , "filt2" , &flt2_path, "output second filtering path"),
		OPT_STRING ( 0 , "smoo1" , &smo1_path, "output smoothing path"),
		OPT_INTEGER('f', "first" , &fframe, "first frame"),
		OPT_INTEGER('l', "last"  , &lframe , "last frame"),
		OPT_FLOAT  ('s', "sigma" , &sigma, "noise standard dev"),

		OPT_GROUP("First filtering options"),
		OPT_INTEGER( 0 , "f1_p"     , &f1_prms.patch_sz, "patch size"),
		OPT_INTEGER( 0 , "f1_sx"    , &f1_prms.search_sz_x, "search radius (spatial filtering)"),
		OPT_INTEGER( 0 , "f1_st"    , &f1_prms.search_sz_t, "search radius (temporal filtering)"),
#ifndef K_SIMILAR_PATCHES
		OPT_FLOAT  ( 0 , "f1_dth"   , &f1_prms.dista_th, "patch distance threshold"),
#else
		OPT_INTEGER( 0 , "f1_nx"    , &f1_prms.npatches_x, "number of similar patches spatial"),
		OPT_INTEGER( 0 , "f1_nt"    , &f1_prms.npatches_t, "number of similar patches kalman"),
		OPT_INTEGER( 0 , "f1_nt_agg", &f1_prms.npatches_tagg, "number of similar patches kalman spatial average"),
#endif
		OPT_FLOAT  ( 0 , "f1_bx"    , &f1_prms.beta_x, "noise multiplier in spatial filtering"),
		OPT_FLOAT  ( 0 , "f1_bt"    , &f1_prms.beta_t, "noise multiplier in kalman filtering"),
		OPT_FLOAT  ( 0 , "f1_l"     , &f1_prms.dista_lambda, "noisy patch weight in patch distance"),

		OPT_GROUP("Second filtering options"),
		OPT_INTEGER( 0 , "f2_p"     , &f2_prms.patch_sz, "patch size"),
		OPT_INTEGER( 0 , "f2_sx"    , &f2_prms.search_sz_x, "search radius (spatial filtering)"),
		OPT_INTEGER( 0 , "f2_st"    , &f2_prms.search_sz_t, "search radius (temporal filtering)"),
#ifndef K_SIMILAR_PATCHES
		OPT_FLOAT  ( 0 , "f2_dth"   , &f2_prms.dista_th, "patch distance threshold"),
#else
		OPT_INTEGER( 0 , "f2_nx"    , &f2_prms.npatches_x, "number of similar patches spatial"),
		OPT_INTEGER( 0 , "f2_nt"    , &f2_prms.npatches_t, "number of similar patches kalman"),
		OPT_INTEGER( 0 , "f2_nt_agg", &f2_prms.npatches_tagg, "number of similar patches kalman spatial average"),
#endif
		OPT_FLOAT  ( 0 , "f2_bx"    , &f2_prms.beta_x, "noise multiplier in spatial filtering"),
		OPT_FLOAT  ( 0 , "f2_bt"    , &f2_prms.beta_t, "noise multiplier in kalman filtering"),
		OPT_FLOAT  ( 0 , "f2_l"     , &f2_prms.dista_lambda, "noisy patch weight in patch distance"),

		OPT_GROUP("Smoothing options"),
		OPT_INTEGER( 0 , "s1_p"     , &s1_prms.patch_sz, "patch size"),
		OPT_INTEGER( 0 , "s1_st"    , &s1_prms.search_sz_t, "search region radius"),
#ifndef K_SIMILAR_PATCHES
		OPT_FLOAT  ( 0 , "s1_dth"   , &s1_prms.dista_th, "patch distance threshold"),
#else
		OPT_INTEGER( 0 , "s1_nt"    , &s1_prms.npatches_t, "number of similar patches kalman"),
		OPT_INTEGER( 0 , "s1_nt_agg", &s1_prms.npatches_tagg, "number of similar patches kalman spatial average"),
#endif
		OPT_FLOAT  ( 0 , "s1_bt"    , &s1_prms.beta_t, "noise multiplier in kalman filtering"),
		OPT_FLOAT  ( 0 , "s1_l"     , &s1_prms.dista_lambda, "noisy patch weight in patch distance"),

		OPT_GROUP("Program options"),
		OPT_BOOLEAN('v', "verbose", &verbose, "verbose output"),
		OPT_END(),
	};

	// parse command line
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nA video denoiser based on non-local means.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// determine mode
	bool second_filt = (f2_prms.patch_sz && (flt2_path || smo1_path));
	bool smoothing   = (s1_prms.patch_sz && smo1_path);

	// check if output paths have been provided
	if ((f1_prms.patch_sz == 0))
		return fprintf(stderr, "Error: f1_p == 0, exiting\n"), 1;

	if (!flt1_path && !(flt2_path && f2_prms.patch_sz) && !smoothing)
		return fprintf(stderr, "Error: no output path given for any "
		                       "computed output - exiting\n"), 1;

	if (!flt1_path && !flt2_path && s1_prms.patch_sz == 0)
		return fprintf(stderr, "Error: s1_p == 0 and no output paths given for "
		                       "filt1 and filt2\n"), 1;

	if (f2_prms.patch_sz == 0 && flt2_path)
		fprintf(stderr, "Warning: f2_p == 0 - no output files will "
		                "be stored in %s\n", flt2_path);

	if (s1_prms.patch_sz == 0 && smo1_path)
		fprintf(stderr, "Warning: s1_p == 0 - no output files will "
		                "be stored in %s\n", smo1_path);

	// default value for noise-dependent params
	nlkalman_default_params(&f1_prms, sigma);
	nlkalman_default_params(&f2_prms, sigma);
	nlkalman_default_params(&s1_prms, sigma);

	// print parameters
	if (verbose)
	{
		printf("data input:\n");
		printf("\tnoise         %5.2f\n", sigma);
		printf("\tfirst frame   %d\n", fframe);
		printf("\tlast frame    %d\n", lframe);
		printf("\tnoisy frames  %s\n", nisy_path);
		printf("\tbwd flows     %s\n", bflo_path);
		printf("\tfwd flows     %s\n", fflo_path);
		printf("\tbwd occlus.   %s\n", bocc_path);
		printf("\tfwd occlus.   %s\n", focc_path);
		printf("\n");

		printf("data output:\n");
		printf("\tfiltering 1   %s\n", flt1_path);
		printf("\tfiltering 2   %s\n", flt2_path);
		printf("\tsmoothing 1   %s\n", smo1_path);
		printf("\n");

		printf("first filtering parameters:\n");
		printf("\tpatch      %d\n", f1_prms.patch_sz);
		printf("\tsearch_x   %d\n", f1_prms.search_sz_x);
		printf("\tsearch_t   %d\n", f1_prms.search_sz_t);
#ifndef K_SIMILAR_PATCHES
		printf("\tdth        %g\n", f1_prms.dista_th);
#else
		printf("\tnp_x       %d\n", f1_prms.npatches_x);
		printf("\tnp_t       %d\n", f1_prms.npatches_t);
		printf("\tnp_tagg    %d\n", f1_prms.npatches_tagg);
#endif
		printf("\tlambda     %g\n", f1_prms.dista_lambda);
		printf("\tbeta_x     %g\n", f1_prms.beta_x);
		printf("\tbeta_t     %g\n", f1_prms.beta_t);
		printf("\n");

		if (second_filt)
		{
			printf("second filtering parameters:\n");
			printf("\tpatch      %d\n", f2_prms.patch_sz);
			printf("\tsearch_x   %d\n", f2_prms.search_sz_x);
			printf("\tsearch_t   %d\n", f2_prms.search_sz_t);
#ifndef K_SIMILAR_PATCHES
			printf("\tdth        %g\n", f2_prms.dista_th);
#else
			printf("\tnp_x       %d\n", f2_prms.npatches_x);
			printf("\tnp_t       %d\n", f2_prms.npatches_t);
			printf("\tnp_tagg    %d\n", f2_prms.npatches_tagg);
#endif
			printf("\tlambda     %g\n", f2_prms.dista_lambda);
			printf("\tbeta_x     %g\n", f2_prms.beta_x);
			printf("\tbeta_t     %g\n", f2_prms.beta_t);
			printf("\n");
		}

		if (smoothing)
		{
			printf("smoothing parameters:\n");
			printf("\tpatch      %d\n", s1_prms.patch_sz);
			printf("\tsearch_t   %d\n", s1_prms.search_sz_t);
#ifndef K_SIMILAR_PATCHES
			printf("\tdth        %g\n", s1_prms.dista_th);
#else
			printf("\tnp_t       %d\n", s1_prms.npatches_t);
			printf("\tnp_tagg    %d\n", s1_prms.npatches_tagg);
#endif
			printf("\tlambda     %g\n", s1_prms.dista_lambda);
			printf("\tbeta_t     %g\n", s1_prms.beta_t);
			printf("\n");
		}
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
#ifdef DECOUPLE_FILTER2
			// instead of using the output of the 2nd step as
			// previous frame, use the output of the 1st step
			float * deno0 = bsic1;
#else
			float * deno0 = deno + (f - 1 - fframe)*whc;
#endif
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
		nlkalman_filter_frame(bsic1, nisy1, deno0, NULL, w, h, c, sigma, f1_prms, f);

		// save output
		if (flt1_path)
		{
			sprintf(frame_name, flt1_path, f);
			iio_save_image_float_vec(frame_name, bsic1, w, h, c);
		}

		// filtering 2nd step [[[3
		if (second_filt && f > fframe)
		{
#ifdef DECOUPLE_FILTER2
			// instead of using the output of the 2nd step as
			// previous frame, use the output of the 1st step
			float * deno0 = deno + (f - 1 - fframe)*whc;

			if (bflo)
			{
				float * flow0 = bflo + (f - fframe)*wh2;
				float * occl1 = bocc ? bocc + (f - fframe)*w*h : NULL;
				warp_bicubic(warp0, deno0, flow0, occl1, w, h, c);
				deno0 = warp0;
			}
#endif
			nlkalman_filter_frame(deno1, nisy1, deno0, bsic1, w, h, c, sigma, f2_prms, f);
			memcpy(nisy1, deno1, whc*sizeof(float));
		}
		else
			memcpy(nisy1, bsic1, whc*sizeof(float));

		// save output
		if (second_filt && flt2_path)
		{
			sprintf(frame_name, flt2_path, f);
			iio_save_image_float_vec(frame_name, nisy1, w, h, c);
		}

		// smoothing [[[3
		if (smoothing && f > fframe)
		{
			// point to previous frame
			float *filt1 = deno + (f - 1 - fframe)*whc;

			// warp current frame to previous frame [[[4
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

			// run smoother [[[4
			nlkalman_smooth_frame(deno1, filt1, smoo0, NULL, w, h, c, sigma, s1_prms, f);
			memcpy(filt1, deno1, whc*sizeof(float));
		}

		// save output
		if (smoothing && f > fframe)
		{
			sprintf(frame_name, smo1_path, f-1);
			iio_save_image_float_vec(frame_name, deno1, w, h, c);
		}
	}


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
