// comment to decouple the 1st filtering stage from the 2nd
#define DECOUPLE_FILTER2

// comment for patch distance using previous frame
//#define LAMBDA_DISTANCE

// comment for uniform aggregation
#define WEIGHTED_AGGREGATION

// comment for distance threshold
#define K_SIMILAR_PATCHES

// opponent color transform
void rgb2opp(float *im, int w, int h, int ch);
void opp2rgb(float *im, int w, int h, int ch);

// warping using a displacement field
void warp_bicubic(float *imw, float *im, float *of, float *msk,
		int w, int h, int ch);

// parameter structure
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

// default parameters as a function of sigma
enum FILTER_MODE {FLT1, FLT2, SMO1};

void nlkalman_default_params(struct nlkalman_params * p, float sigma,
		enum FILTER_MODE mode);

// nl-kalman filtering of a frame (with k similar patches)
void nlkalman_filter_frame(float *deno1, float *nisy1, float *deno0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame);

// nl-kalman smoothing of a frame (with k similar patches)
void nlkalman_smooth_frame(float *smoo1, float *filt1, float *smoo0, float *bsic1,
		int w, int h, int ch, float sigma,
		const struct nlkalman_params prms, int frame);
