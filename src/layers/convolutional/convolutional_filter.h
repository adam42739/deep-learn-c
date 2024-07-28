#pragma once
#include "../linear/linear_layer.h"
#include "image_structures.h"

typedef struct ConvolutionalFilter
{
	LinearModel* linmod;
	int m;
	int n;
	int* img_array_indexes;
	int num_indexes;
} ConvolutionalFilter;

ConvolutionalFilter* cnn_filter_alloc(int m, int n, int num_indexes, int* indexes);

void cnn_filter_free(ConvolutionalFilter* cnn_filter);

void cnn_filter_randomize_weights(ConvolutionalFilter* cnn_filter, __rng_dist_type rng_dist);

void cnn_filter_set_bias_zero(ConvolutionalFilter* cnn_filter);

double cnn_filter_forward_at_index(ConvolutionalFilter* cnn_filter, ImageLayer* input, int i_i, int j_i);

void cnn_filter_forward(ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* output);