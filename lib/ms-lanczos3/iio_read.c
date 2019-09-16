// octave/iio interface

#include <mex.h>
#include "iio.h"

// x = iio_read("lena.tiff");
// [x,y,z] = iio_read("lena.tiff", "barbara.tiff", "baboon.png");
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nlhs != nrhs)
		return;

	int i, j, k;
	for (i = 0; i < nrhs; i++)
	{
		char *f = mxArrayToString(prhs[i]);
		int w, h, d;
		double *x = iio_read_image_double_vec(f, &w, &h, &d);
		mxFree(f);
		if (!x) continue;
		mwSize S[3] = {h, w, d};
		plhs[i] = mxCreateNumericArray(3, S, mxDOUBLE_CLASS, mxREAL);
		double *X = mxGetPr(plhs[i]);
		for (j = 0; j < h; j++)
		for (i = 0; i < w; i++)
		for (k = 0; k < d; k++)
			X[w*h*k + (i*h+j)] = x[(j*w+i)*d + k];
		iio_free(x);
	}
}
