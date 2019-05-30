#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

#include <stdio.h>     // getchar() for debugging

#include "nlkalman.h"

// 'usage' message in the command line
static const char *const usages[] = {
	"nlkalman-flt [options] [[--] args]",
	"nlkalman-flt [options]",
	NULL,
};

// frame-by-frame filtering main
int main(int argc, const char *argv[])
{
//	omp_set_num_threads(2);
	// parse command line [[[2

	// command line parameters and their defaults
	const char *noisy_path = NULL; // input noisy frame path
	const char *bflow_path = NULL; // input bwd flow path
	const char *boccl_path = NULL; // input bwd occlusion path
	const char *flt10_path = NULL; // input previous first filtering path
	const char *flt20_path = NULL; // input previous second filtering path
	const char *flt11_path = NULL; // output first filtering path
	const char *flt21_path = NULL; // output second filtering path
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

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Data i/o options"),
		OPT_STRING ('i', "nisy" , &noisy_path, "input noisy frames path"),
		OPT_STRING ('o', "bflo" , &bflow_path, "input bwd flow path"),
		OPT_STRING ('k', "bocc" , &boccl_path, "input bwd occlusion masks path"),
		OPT_STRING ( 0 , "flt10", &flt10_path, "input previous first filtering path"),
		OPT_STRING ( 0 , "flt20", &flt20_path, "input previous second filtering path"),
		OPT_STRING ( 0 , "flt11", &flt11_path, "input/output first filtering path"),
		OPT_STRING ( 0 , "flt21", &flt21_path, "output second filtering path"),
		OPT_FLOAT  ('s', "sigma", &sigma, "noise standard dev"),

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

		OPT_GROUP("Program options"),
//		OPT_BOOLEAN('v', "verbose", &verbose    , "verbose output"),
		OPT_INTEGER('v', "verbose", &verbose_int, "verbose output"),
		OPT_END(),
	};

	// parse command line
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nPatch-based Kalman filter for video denoising.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// hack around argparse bug
	verbose = (bool)verbose_int;

	// determine mode
	bool apply_filt1 = (f1_prms.patch_sz);
	bool apply_filt2 = (f2_prms.patch_sz && flt21_path);

	// check if output paths have been provided
	if (!apply_filt1 && !apply_filt2)
		return fprintf(stderr, "Error: nothing to do, exiting\n"), 1;

	if (!apply_filt1 && !flt11_path)
		return fprintf(stderr, "Error: f1_p == 0 and no input path given, exiting\n"), 1;

	if (!flt11_path && !apply_filt2)
		return fprintf(stderr, "Error: no output path given for any "
		                       "computed output - exiting\n"), 1;

	if (!flt11_path && !flt21_path)
		return fprintf(stderr, "Error: s1_p == 0 and no output paths given for "
		                       "filt1 and filt2\n"), 1;

	if (f2_prms.patch_sz == 0 && flt21_path)
		fprintf(stderr, "Warning: f2_p == 0 - no output files will "
		                "be stored in %s\n", flt21_path);

	// default value for noise-dependent params
	nlkalman_default_params(&f1_prms, sigma, FLT1);
	nlkalman_default_params(&f2_prms, sigma, FLT2);

	// print parameters
	if (verbose)
	{
		printf("data input:\n");
		printf("\tnoise         %05.2f\n", sigma);
		printf("\tnoisy frames  %s\n", noisy_path);
		printf("\tbwd flows     %s\n", bflow_path);
		printf("\tbwd occlus.   %s\n", boccl_path);
		printf("\tprev filt 1   %s\n", flt10_path);
		printf("\tprev filt 2   %s\n", flt20_path);
		if (!apply_filt1)
			printf("\tfiltering 1   %s\n", flt11_path);
		printf("\n");

		printf("data output:\n");
		if (apply_filt1)
			printf("\tfiltering 1   %s\n", flt11_path);
		printf("\tfiltering 2   %s\n", flt21_path);
		printf("\n");

		if (apply_filt1)
		{
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
		}

		if (apply_filt2)
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
	}

	// load data [[[2
	int w, h, c;
	float *nisy = iio_read_image_float_vec(noisy_path, &w, &h, &c);
	if (!nisy)
		return fprintf(stderr, "Error while openning bwd optical flow\n"), 1;

	// load backward optical flow [[[3
	float * bflo = NULL;
	if (bflow_path)
	{
		int w1, h1, c1;
		bflo = iio_read_image_float_vec(bflow_path, &w1, &h1, &c1);

		if (!bflo)
		{
			if (nisy) free(nisy);
			return fprintf(stderr, "Error while openning bwd optical flow\n"), 1;
		}

		if (w*h != w1*h1 || c1 != 2)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			return fprintf(stderr, "Frame and optical flow size missmatch\n"), 1;
		}
	}

	// load backward occlusion masks [[[3
	float * bocc = NULL;
	if (bflow_path && boccl_path)
	{
		int w1, h1, c1;
		bocc = iio_read_image_float_vec(boccl_path, &w1, &h1, &c1);

		if (!bocc)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			return fprintf(stderr, "Error while openning occlusion mask\n"), 1;
		}

		if (w*h != w1*h1 || c1 != 1)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (bocc) free(bocc);
			return fprintf(stderr, "Frame and occlusion mask size missmatch\n"), 1;
		}
	}

	// load filter 1 output from previous frame [[[3
	float * flt10 = NULL;
	if (flt10_path)
	{
		int w1, h1, c1;
		flt10 = iio_read_image_float_vec(flt10_path, &w1, &h1, &c1);

		if (!flt10)
			fprintf(stderr, "Error while openning previous filter 1 output\n");

		if (flt10 && w*h*c != w1*h1*c1)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (bocc) free(bocc);
			if (flt10) free(flt10);
			return fprintf(stderr, "Frame and previous filter 1 output size missmatch\n"), 1;
		}
	}

	// load filter 2 output from previous frame [[[3
	float * flt20 = NULL;
	if (flt20_path)
	{
		int w1, h1, c1;
		flt20 = iio_read_image_float_vec(flt20_path, &w1, &h1, &c1);

		if (!flt20)
			fprintf(stderr, "Error while openning previous filter 2 output\n");

		if (flt20 && w*h*c != w1*h1*c1)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (bocc) free(bocc);
			if (flt10) free(flt10);
			if (flt20) free(flt20);
			return fprintf(stderr, "Frame and previous filter 2 output size missmatch\n"), 1;
		}
	}

	// load filter 1 output from current frame [[[3
	float * flt11 = NULL;
	if (!apply_filt1)
	{
		int w1, h1, c1;
		flt11 = iio_read_image_float_vec(flt11_path, &w1, &h1, &c1);

		if (!flt11)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (bocc) free(bocc);
			if (flt10) free(flt10);
			if (flt20) free(flt20);
			return fprintf(stderr, "Error while openning filter 1 output\n"), 1;
		}

		if (flt11 && w*h*c != w1*h1*c1)
		{
			if (nisy) free(nisy);
			if (bflo) free(bflo);
			if (bocc) free(bocc);
			if (flt10) free(flt10);
			if (flt20) free(flt20);
			if (flt11) free(flt11);
			return fprintf(stderr, "Frame and filter 1 output size missmatch\n"), 1;
		}
	}


	// run denoiser - forward pass [[[2
	const int whc = w*h*c, wh2 = w*h*2;
	float * warp0 = malloc(whc*sizeof(float));

	// change color space
	rgb2opp(nisy, w, h, c);
	if (flt10) rgb2opp(flt10, w, h, c);
	if (flt20) rgb2opp(flt20, w, h, c);

	// 1st filtering step [[[3
	if (apply_filt1)
	{
		// warp previous denoised frame
		if (flt10 && bflo)
		{
			warp_bicubic(warp0, flt10, bflo, bocc, w, h, c);
			float *tmp = flt10; flt10 = warp0; warp0 = tmp; // swap warp0-filt10
		}

		// filter
		flt11 = malloc(whc*sizeof(float));
		nlkalman_filter_frame(flt11, nisy, flt10, NULL, w, h, c, sigma, f1_prms, 0);
	}
	else rgb2opp(flt11, w, h, c);

	// 2nd filtering step [[[3
	float * flt21 = NULL;
	if (apply_filt2)
	{
		// warp previous denoised frame
		if (bflo && flt20)
		{
			warp_bicubic(warp0, flt20, bflo, bocc, w, h, c);
			float *tmp = flt20; flt20 = warp0; warp0 = tmp; // swap warp0-filt20
		}

		// filter
		flt21 = malloc(whc*sizeof(float));
		nlkalman_filter_frame(flt21, nisy, flt20, flt11, w, h, c, sigma, f2_prms, 0);

		// save second filtering output
		if (flt11_path)
		{
			opp2rgb(flt21, w, h, c);
			iio_save_image_float_vec(flt21_path, flt21, w, h, c);
		}
	}

	// save first filtering output
	if (apply_filt1 && flt11_path)
	{
		opp2rgb(flt11, w, h, c);
		iio_save_image_float_vec(flt11_path, flt11, w, h, c);
	}

	if (flt21) free(flt21);
	if (flt11) free(flt11);
	if (warp0) free(warp0);
	if (flt10) free(flt10);
	if (flt20) free(flt20);
	if (nisy) free(nisy);
	if (bflo) free(bflo);
	if (bocc) free(bocc);

	return EXIT_SUCCESS; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
