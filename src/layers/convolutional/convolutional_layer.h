#include "convolutional_filter.h"
#include "image_structures.h"

typedef struct ConvolutionalLayer
{
	ConvolutionalFilter **filters;
	int num_filters;
} ConvolutionalLayer;

ConvolutionalLayer *cnn_layer_alloc(int num_filters, int m, int n);

void cnn_layer_free(ConvolutionalLayer* cnn_layer);

void cnn_layer_randomize_weights(ConvolutionalLayer* cnn_layer, __rng_dist_type rng_dist);

void cnn_layer_set_bias_zero(ConvolutionalLayer* cnn_layer);

typedef struct GreyScale
{
	double *white;
	int m;
	int n;
} GreyScale;

typedef struct Image
{
	double *R;
	double *G;
	double *B;
	int m;
	int n;
} Image;
