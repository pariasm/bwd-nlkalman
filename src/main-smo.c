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
	"nlkalman-smo [options] [[--] args]",
	"nlkalman-smo [options]",
	NULL,
};

// frame-by-frame smoothing main
int main(int argc, const char *argv[])
{
	omp_set_num_threads(2);
	// parse command line [[[2

	// command line parameters and their defaults [[[3
	const char *flt1_path = NULL; // input filtered frame path
	const char *smo0_path = NULL; // input previous filtered frame path
	const char *fflo_path = NULL; // input fwd flow path
	const char *focc_path = NULL; // input fwd occlusion path
	const char *smo1_path = NULL; // output smoothing path
	float sigma = 0.f;
	bool verbose = false;
	int  verbose_int = 0; // hack around bug in argparse

	// smoothing options
	struct nlkalman_params s1_prms;
	s1_prms.patch_sz      = -1; // -1 means automatic value
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

	// configure command line parser [[[3
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Data i/o options"),
		OPT_STRING ( 0 , "flt1", &flt1_path, "input filtered frame path"),
		OPT_STRING ( 0 , "smo0", &smo0_path, "input next smoothed frame path"),
		OPT_STRING ('o', "fflo", &fflo_path, "input fwd flow path"),
		OPT_STRING ('k', "focc", &focc_path, "input fwd occlusion mask path"),
		OPT_STRING ( 0 , "smo1", &smo1_path, "output smoothed frame"),
		OPT_FLOAT  ('s', "sigma" , &sigma, "noise standard dev"),

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
//		OPT_BOOLEAN('v', "verbose", &verbose    , "verbose output"),
		OPT_INTEGER('v', "verbose", &verbose_int, "verbose output"),
		OPT_END(),
	};

	// parse command line [[[3
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nPatch-based Kalman smoother for video denoising.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// hack around argparse bug
	verbose = (bool)verbose_int;

	// error checking
	if (!smo1_path) return fprintf(stderr, "Error: no output path given\n"), 1;
	if (s1_prms.patch_sz == 0) return fprintf(stderr, "Error: s1_p == 0\n"), 1;

	// default value for noise-dependent params [[[3
	nlkalman_default_params(&s1_prms, sigma, SMO1);

	// print parameters
	if (verbose)
	{
		printf("data input:\n");
		printf("\tnoise         %05.2f\n", sigma);
		printf("\tfiltering 1   %s\n", flt1_path);
		printf("\tfiltering 0   %s\n", smo0_path);
		printf("\tfwd flows     %s\n", fflo_path);
		printf("\tfwd occlus.   %s\n", focc_path);
		printf("\n");

		printf("data output:\n");
		printf("\tsmoothing 1   %s\n", smo1_path);
		printf("\n");

		printf("smoother params:\n");
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

	// load data [[[2

	// load filtered frame [[[3
	int w, h, c;
	float * flt1 = iio_read_image_float_vec(flt1_path, &w, &h, &c);
	if (!flt1) return fprintf(stderr, "Opening %s failed\n", flt1_path), 1;

	// load next filtered frame [[[3
	int w1, h1, c1;
	float * smo0 = iio_read_image_float_vec(smo0_path, &w1, &h1, &c1);
	if (!smo0) return fprintf(stderr, "Opening %s failed\n", smo0_path), 1;
	if (w*h*c != w1*h1*c1)
	{
		if (flt1) free(flt1);
		if (smo0) free(smo0);
		return fprintf(stderr, "Filtered frames size missmatch\n"), 1;
	}

	// load forward optical flow [[[3
	float * fflo = NULL;
	if (fflo_path)
	{
		int w1, h1, c1;
		fflo = iio_read_image_float_vec(fflo_path, &w1, &h1, &c1);

		if (!fflo)
		{
			if (flt1) free(flt1);
			if (smo0) free(smo0);
			return fprintf(stderr, "Opening %s failed\n", fflo_path), 1;
		}

		if (w*h != w1*h1 || c1 != 2)
		{
			if (flt1) free(flt1);
			if (smo0) free(smo0);
			if (fflo) free(fflo);
			return fprintf(stderr, "Frame and optical flow size missmatch\n"), 1;
		}
	}

	// load forward occlusion masks [[[3
	float * focc = NULL;
	if (fflo_path && focc_path)
	{
		int w1, h1, c1;
		focc = iio_read_image_float_vec(focc_path, &w1, &h1, &c1);

		if (!focc)
		{
			if (flt1) free(flt1);
			if (smo0) free(smo0);
			if (fflo) free(fflo);
			return fprintf(stderr, "Opening %s failed\n", focc_path), 1;
		}

		if (w*h != w1*h1 || c1 != 1)
		{
			if (flt1) free(flt1);
			if (smo0) free(smo0);
			if (fflo) free(fflo);
			if (focc) free(focc);
			return fprintf(stderr, "Frame and occlusion mask size missmatch\n"), 1;
		}
	}

	// run smoother [[[2
	const int whc = w*h*c;
	float * wrp0 = malloc(whc*sizeof(float));
	float * smo1 = malloc(whc*sizeof(float));

	// change color space
	rgb2opp(flt1, w, h, c);
	rgb2opp(smo0, w, h, c);

	// warp next frame to current frame
	if (fflo)
	{
		warp_bicubic(wrp0, smo0, fflo, focc, w, h, c);
		float *tmp = smo0; smo0 = wrp0; wrp0 = tmp; // swap wrp0-smo0
	}

	// run smoother
	nlkalman_smooth_frame(smo1, flt1, smo0, NULL, w, h, c, sigma, s1_prms, 0);

	// save first filtering output
	opp2rgb(smo1, w, h, c);
	iio_save_image_float_vec(smo1_path, smo1, w, h, c);

	if (wrp0) free(wrp0);
	if (smo1) free(smo1);
	if (smo0) free(smo0);
	if (flt1) free(flt1);
	if (fflo) free(fflo);
	if (focc) free(focc);

	return 1; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
