#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

#include <stdio.h>     // getchar() for debugging

#include "nlkalman.h"

// read/write image sequence [[[1
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

// main [[[1

// 'usage' message in the command line
static const char *const usages[] = {
	"nlkalman-seq [options] [[--] args]",
	"nlkalman-seq [options]",
	NULL,
};

int main(int argc, const char *argv[])
{
	omp_set_num_threads(2);
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
	int  verbose_int = 0; // hack around bug in argparse

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
	bool full_smoother = true; // next frame smoother or full video smoother
	struct nlkalman_params s1_prms;
	s1_prms.patch_sz      = 0; // -1 means automatic value
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
		OPT_BOOLEAN( 0 , "s1_full"  , &full_smoother, "next frame (default) or full video smoothing"),

		OPT_GROUP("Program options"),
//		OPT_BOOLEAN('v', "verbose", &verbose    , "verbose output"),
		OPT_INTEGER('v', "verbose", &verbose_int, "verbose output"),
		OPT_END(),
	};

	// parse command line
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nA video denoiser based on non-local means.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// hack around argparse bug
	verbose = (bool)verbose_int;

	// determine mode
	bool second_filt = (f2_prms.patch_sz && (flt2_path || smo1_path));
	bool next_frame_smoother = (!full_smoother && s1_prms.patch_sz && smo1_path);
	full_smoother = (full_smoother && s1_prms.patch_sz && smo1_path);

	// check if output paths have been provided
	if ((f1_prms.patch_sz == 0))
		return fprintf(stderr, "Error: f1_p == 0, exiting\n"), 1;

	if (!flt1_path && !(flt2_path && f2_prms.patch_sz) && !next_frame_smoother)
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
	nlkalman_default_params(&f1_prms, sigma, FLT1);
	nlkalman_default_params(&f2_prms, sigma, FLT2);
	nlkalman_default_params(&s1_prms, sigma, SMO1);

	// print parameters
	if (verbose)
	{
		printf("data input:\n");
		printf("\tnoise         %05.2f\n", sigma);
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

		if (next_frame_smoother || full_smoother)
		{
			printf("%s smoother params:\n", full_smoother? "full": "single frame");
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

	// run denoiser - forward pass [[[2
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
		rgb2opp(nisy1, w, h, c);
		nlkalman_filter_frame(bsic1, nisy1, deno0, NULL, w, h, c, sigma, f1_prms, f);

		// save output
		if (flt1_path)
		{
			sprintf(frame_name, flt1_path, f);
			opp2rgb(bsic1, w, h, c);
			iio_save_image_float_vec(frame_name, bsic1, w, h, c);
			rgb2opp(bsic1, w, h, c);
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
			opp2rgb(nisy1, w, h, c);
			iio_save_image_float_vec(frame_name, nisy1, w, h, c);
			rgb2opp(nisy1, w, h, c);
		}

		// smoothing [[[3
		if (next_frame_smoother && f > fframe)
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
		if (next_frame_smoother && f > fframe)
		{
			sprintf(frame_name, smo1_path, f-1);
			opp2rgb(deno1, w, h, c);
			iio_save_image_float_vec(frame_name, deno1, w, h, c);
//			rgb2opp(deno1, w, h, c);
		}
	}

	// run full video smoother - backward pass [[[2
	if (full_smoother)
	for (int f = lframe-1; f >= fframe; --f)
	{
		if (verbose) printf("processing frame %d\n", f);

		// warp next frame to current frame [[[4
		float * smoo0 = deno + (f + 1 - fframe)*whc;
		{
			if (fflo)
			{
				float * flow0 = fflo + (f - fframe)*wh2;
				float * occl0 = focc ? focc + (f - fframe)*w*h : NULL;
				warp_bicubic(warp0, smoo0, flow0, occl0, w, h, c);
			}
			else
				// copy without warping
				memcpy(warp0, smoo0, whc*sizeof(float));

			smoo0 = warp0;
		}

		// run smoother [[[4
		float * filt1 = deno + (f - fframe)*whc;
		nlkalman_smooth_frame(deno1, filt1, smoo0, NULL, w, h, c, sigma, s1_prms, f);
		memcpy(filt1, deno1, whc*sizeof(float));

		// save output
		sprintf(frame_name, smo1_path, f);
		opp2rgb(deno1, w, h, c);
		iio_save_image_float_vec(frame_name, deno1, w, h, c);
		rgb2opp(deno1, w, h, c);
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
