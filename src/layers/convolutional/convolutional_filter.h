#pragma once
#include "../linear/linear_layer.h"
#include "image_structures.h"

typedef struct ConvolutionalFilter
{
	LinearModel* linmod;
	int filter_m;
	int filter_n;
	double bias;
	int* img_array_indexes;
	int num_indexes;
} ConvolutionalFilter;

ConvolutionalFilter* cnn_filter_alloc(int filter_m, int filter_n, int num_indexes, int* indexes);

void cnn_filter_free(ConvolutionalFilter* cnn_filter);

void cnn_filter_randomize_weights(ConvolutionalFilter* cnn_filter, __rng_dist_type rng_dist);

void cnn_filter_set_bias_zero(ConvolutionalFilter* cnn_filter);

double cnn_filter_forward_at_index(ConvolutionalFilter* cnn_filter, ImageLayer* input, int i_i, int j_i);

void cnn_filter_forward(ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* output);

typedef struct ConvolutionalFilterGrad
{
	LinearModelGrad** linmod_grads;
	int output_m;
	int output_n;
} ConvolutionalFilterGrad;

ConvolutionalFilterGrad* cnn_filter_grad_alloc(ConvolutionalFilter* cnn_filter, int output_m, int output_n);

void cnn_filter_grad_free(ConvolutionalFilterGrad* cnn_filter_grad);

void cnn_filter_grad_compute_weights(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net);

void cnn_filter_grad_compute_bias(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net);

void cnn_filter_grad_compute_input(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net);

void cnn_filter_grad_compute(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net);