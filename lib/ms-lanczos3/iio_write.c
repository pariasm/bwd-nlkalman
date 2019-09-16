// octave/iio interface

#include <mex.h>
#include <stdlib.h>
#include "iio.h"

// iio_write("lena.tiff", x)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nlhs != 0 || nrhs != 2)
		return;

	const mwSize *dims = mxGetDimensions(prhs[1]);
	int ndims = mxGetNumberOfDimensions(prhs[1]);
	if (ndims >= 3) ndims = 3;

	int h = dims[0];
	int w = ndims > 1 ? dims[1] : 1;
	int d = ndims > 2 ? dims[2] : 1;

	char *f = mxArrayToString(prhs[0]);

	int i,j,k;

	switch(mxGetClassID(prhs[1]))
	{
		case mxDOUBLE_CLASS:
		{
			double *X = mxGetPr(prhs[1]);
			double *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = X[w*h*k + (i*h+j)];
			iio_write_image_double_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxSINGLE_CLASS:
		{
			float *X = (float*) mxGetPr(prhs[1]);
			float *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = X[w*h*k + (i*h+j)];
			iio_write_image_float_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxINT8_CLASS:
		{
			int8_t *X = (int8_t*) mxGetPr(prhs[1]);
			int *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = (int)X[w*h*k + (i*h+j)];
			iio_write_image_int_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxINT16_CLASS:
		{
			int16_t *X = (int16_t*) mxGetPr(prhs[1]);
			int *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = (int)X[w*h*k + (i*h+j)];
			iio_write_image_int_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxINT32_CLASS:
		{
			int32_t *X = (int32_t*) mxGetPr(prhs[1]);
			int *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = (int)X[w*h*k + (i*h+j)];
			iio_write_image_int_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxUINT8_CLASS:
		{
			uint8_t *X = (uint8_t*) mxGetPr(prhs[1]);
			uint8_t *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = X[w*h*k + (i*h+j)];
			iio_write_image_uint8_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxUINT16_CLASS:
		{
			uint16_t *X = (uint16_t*) mxGetPr(prhs[1]);
			uint16_t *x = malloc(w*h*d*sizeof*x);
			for (j = 0; j < h; j++)
			for (i = 0; i < w; i++)
			for (k = 0; k < d; k++)
				x[(j*w+i)*d + k] = X[w*h*k + (i*h+j)];
			iio_write_image_uint16_vec(f, x, w, h, d);
			free(x);
			break;
		}

		case mxLOGICAL_CLASS:
		case mxUINT32_CLASS:
		case mxINT64_CLASS:
		case mxUINT64_CLASS:
			mexErrMsgTxt("Classes Logical, uint32, int64 and uint64 not "
					"handled. Try with a more common one.");
			break;

		case mxUNKNOWN_CLASS:
		case mxCELL_CLASS:
		case mxSTRUCT_CLASS:
		case mxCHAR_CLASS:
//		case mxVOID_CLASS:
		case mxFUNCTION_CLASS:
			mexErrMsgTxt("Invalid class.");
			break;

		default:
			mexErrMsgTxt("Class not recognized.");
	}
	mxFree(f);
}
