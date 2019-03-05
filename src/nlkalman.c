#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

#include <stdio.h>     // getchar() for debugging

#include "nlkalman.h"

// some macros [[[1

#define max(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a > _b ? _a : _b; })

#define min(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

// bicubic interpolation [[[1

#ifdef NAN
// extrapolate by nan
float getsample_nan(float *x, int w, int h, int pd, int i, int j, int l)
{
	assert(l >= 0 && l < pd);
	return (i < 0 || i >= w || j < 0 || j >= h) ? NAN : x[(i + j*w)*pd + l];
}
#endif//NAN

float cubic_interpolation(float v[4], float x)
{
	return v[1] + 0.5 * x*(v[2] - v[0]
			+ x*(2.0*v[0] - 5.0*v[1] + 4.0*v[2] - v[3]
			+ x*(3.0*(v[1] - v[2]) + v[3] - v[0])));
}

float bicubic_interpolation_cell(float p[4][4], float x, float y)
{
	float v[4];
	v[0] = cubic_interpolation(p[0], y);
	v[1] = cubic_interpolation(p[1], y);
	v[2] = cubic_interpolation(p[2], y);
	v[3] = cubic_interpolation(p[3], y);
	return cubic_interpolation(v, x);
}

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

// opponent color transform [[[1

void rgb2opp(float *im, int w, int h, int ch)
{
	if (ch != 3) return;

	const int wh = w*h;
	const float a = 1.f / sqrtf(3.f);
	const float b = 1.f / sqrtf(2.f);
	const float c = 2.f * a * sqrtf(2.f);
	float *p_im = im;
	for (int k = 0; k < wh; ++k, p_im += 3)
	{
		float Y = a * (      p_im[0] +      p_im[1] +         p_im[2]);
		float U = b * (      p_im[0]                -         p_im[2]);
		float V = c * (0.25f*p_im[0] - 0.5f*p_im[1] + 0.25f * p_im[2]);
		p_im[0] = Y;
		p_im[1] = U;
		p_im[2] = V;
	}
}

void opp2rgb(float *im, int w, int h, int ch)
{
	if (ch != 3) return;

	const int wh = w*h;
	const float a = 1.f / sqrtf(3.f);
	const float b = 1.f / sqrtf(2.f);
	const float c = a / b;
	float *p_im = im;
	for (int k = 0; k < wh; ++k, p_im += 3)
	{
		float R = a * p_im[0] + b * p_im[1] + 0.5f * c * p_im[2];
		float G = a * p_im[0]               -        c * p_im[2];
		float B = a * p_im[0] - b * p_im[1] + 0.5f * c * p_im[2];
		p_im[0] = R;
		p_im[1] = G;
		p_im[2] = B;
	}
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




// parameters [[[1

void nlkalman_default_params(struct nlkalman_params * p, float sigma,
		enum FILTER_MODE mode)
{
//	// DERFHD_PARAMS - single filtering step, with distance threshold
//	/* trained for grayscale using
//	 * - derfhd: videos of half hd resolution obtained by downsampling
//	 *           some hd videos from the derf database
//	 *
//	 * the relevant parameters were the patch distance threshold and the b_t
//	 * coefficient that controls the amount of temporal averaging. */
//
//	if (p->patch_sz      < 0) p->patch_sz      = 8;  // not tuned
//	if (p->search_sz_x   < 0) p->search_sz_x   = 10; // not tuned
//	if (p->search_sz_t   < 0) p->search_sz_t   = 10; // not tuned
//	if (p->dista_th      < 0) p->dista_th      = .5*sigma + 15.0;
//	if (p->dista_lambda  < 0) p->dista_lambda  = 1.0;
//	if (p->beta_x        < 0) p->beta_x        = 3.0;
//	if (p->beta_t        < 0) p->beta_t        = 0.05*sigma + 6.0;

	/* TRAIN14_PARAMS 
	 * - two filtering steps + full smoothing
	 * - using k-nn in all steps
	 * 
	 * Trained using 12 sequences from the DAVIS test-challenge dataset plus 2
	 * sequences from DERFHD (old_town_cross and snow_mnt) which have a much
	 * more steady motion. The sequences were dowscaled to 960x540, cropped to
	 * 400x400 and converted to grayscale by channel averaging. Each has 20
	 * frames. The PSNR was computed for the last 10 frames and with 10 pixel
	 * border of the frame removed. */

	if (p->patch_sz      < 0) p->patch_sz      = 8;   // not tuned
	if (p->search_sz_x   < 0) p->search_sz_x   = 10;  // not tuned
	if (p->search_sz_t   < 0) p->search_sz_t   =  5;  // not tuned
	if (p->dista_lambda  < 0) p->dista_lambda  = 1.0; // not tuned

	switch (mode)
	{
		case FLT1:
			if (p->npatches_x    < 0) p->npatches_x    = (int)(0.5*sigma + 40.);
			if (p->beta_x        < 0) p->beta_x        = -0.04*sigma + 3.91;
			if (p->npatches_t    < 0) p->npatches_t    = 30;
			if (p->npatches_tagg < 0) p->npatches_tagg = 20;
			if (p->beta_t        < 0) p->beta_t        = -0.005*sigma + 2.05;
			break;

		case FLT2:
			if (p->npatches_x    < 0) p->npatches_x    = (int)(0.5*sigma + 10.);
			if (p->beta_x        < 0) p->beta_x        = 0.004*sigma + 0.21;
			if (p->npatches_t    < 0) p->npatches_t    = (int)(max(5,sigma));
			if (p->npatches_tagg < 0) p->npatches_tagg = 1;
			if (p->beta_t        < 0) p->beta_t        = 0.014*sigma + 1.38;
			break;

		case SMO1:
			if (p->npatches_x    < 0) p->npatches_x    = 0;
			if (p->beta_x        < 0) p->beta_x        = 0;
			if (p->npatches_t    < 0) p->npatches_t    = (int)(max(5, 3*sigma - 15));
			if (p->npatches_tagg < 0) p->npatches_tagg = p->npatches_t;
			if (p->beta_t        < 0) p->beta_t        = max(1.0, -0.14*sigma + 8.0);
			break;
	}
}

// nl-kalman filtering [[[1

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
		if (aggr1[i] > 1e-6) for (int c = 0; c < ch; ++c, ++j) deno1[j]/= aggr1[i];
		else                 for (int c = 0; c < ch; ++c, ++j) deno1[j] = nisy1[j];

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
		if (aggr1[i] > 1e-6) for (int c = 0; c < ch; ++c, ++j) deno1[j]/= aggr1[i];
		else                 for (int c = 0; c < ch; ++c, ++j) deno1[j] = nisy1[j];

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
		if (aggr1[i] > 1e-6) for (int c = 0; c < ch; ++c, ++j) smoo1[j]/= aggr1[i];
		else                 for (int c = 0; c < ch; ++c, ++j) smoo1[j] = filt1[j];

	// free allocated mem and quit
	dct_threads_destroy(dcts);
	dct_threads_destroy(dcts_pg);
	if (aggr1) free(aggr1);
	if (mask1) free(mask1);

	return; // ]]]2
}


// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
