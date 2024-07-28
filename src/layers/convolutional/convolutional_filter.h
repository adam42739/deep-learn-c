#include "../linear/linear_layer.h"

typedef struct ConvolutionalFilter
{
	LinearModel* linmod;
	int m;
	int n;
} ConvolutionalFilter;

ConvolutionalFilter* cnn_filter_alloc(int m, int n);

void cnn_filter_free(ConvolutionalFilter* cnn_filter);

void cnn_filter_randomize_weights(ConvolutionalFilter* cnn_filter, __rng_dist_type rng_dist);

void cnn_filter_set_bias_zero(ConvolutionalFilter* cnn_filter);