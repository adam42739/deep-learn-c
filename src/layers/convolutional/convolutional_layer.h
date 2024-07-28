#pragma once
#include "convolutional_filter.h"
#include "image_structures.h"

typedef struct ConvolutionalLayer
{
	int num_input_img;
	int input_m;
	int input_n;
	ConvolutionalFilter** filters;
	int num_filters;
	int output_m;
	int output_n;
	__activation_type act_type;
} ConvolutionalLayer;

ConvolutionalLayer* cnn_layer_alloc(
	int num_input_img,
	int input_m,
	int input_n,
	int num_filters,
	int filter_m,
	int filter_n,
	int** indexes,
	int* num_indexes,
	__activation_type act_type
);

void cnn_layer_free(ConvolutionalLayer* cnn_layer);

void cnn_layer_randomize_weights(ConvolutionalLayer* cnn_layer, __rng_dist_type rng_dist);

void cnn_layer_set_bias_zero(ConvolutionalLayer* cnn_layer);

void cnn_layer_forward(ConvolutionalLayer* cnn_layer, ImageLayer* input, ImageLayer* pre_activation, ImageLayer* output);

typedef struct ConvolutionalLayerEvaluation
{
	ImageLayer* pre_activation;
	ImageLayer* output;
} ConvolutionalLayerEvaluation;

ConvolutionalLayerEvaluation* cnn_layer_eval_alloc(ConvolutionalLayer* cnn_layer);

void cnn_layer_eval_free(ConvolutionalLayerEvaluation* cnn_layer_eval);

void cnn_layer_eval_compute(ConvolutionalLayerEvaluation* cnn_layer_eval, ConvolutionalLayer* cnn_layer, ImageLayer* input);

typedef struct ConvolutionalLayerGrad
{
	ConvolutionalFilterGrad** cnn_filter_grads;
	int num_filters;
	ImageLayer* grad_loss_net;
	ImageLayer* grad_out_net;
	ImageLayer* grad_loss_input;
} ConvolutionalLayerGrad;

ConvolutionalLayerGrad* cnn_layer_grad_alloc(ConvolutionalLayer* cnn_layer);

void cnn_layer_grad_free(ConvolutionalLayerGrad* cnn_layer_grad);

void cnn_layer_grad_compute_net(
	ConvolutionalLayerGrad* cnn_layer_grad,
	ConvolutionalLayer* cnn_layer,
	ConvolutionalLayerEvaluation* cnn_layer_eval,
	ImageLayer* grad_loss_out
);

void cnn_layer_grad_compute_filters(ConvolutionalLayerGrad* cnn_layer_grad, ConvolutionalLayer* cnn_layer, ImageLayer* input);

void cnn_layer_grad_compute_input(ConvolutionalLayerGrad* cnn_layer_grad, ConvolutionalLayer* cnn_layer, ImageLayer* grad_loss_out);

void cnn_layer_grad_compute(
	ConvolutionalLayerGrad* cnn_layer_grad,
	ConvolutionalLayer* cnn_layer,
	ConvolutionalLayerEvaluation* cnn_layer_eval,
	ImageLayer* input,
	ImageLayer* grad_loss_out
);
