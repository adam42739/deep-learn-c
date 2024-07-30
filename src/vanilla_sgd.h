#pragma once
#include "layers/linear/linear_layer.h"
#include "layers/convolutional/convolutional_layer.h"

typedef struct LinearLayerSGD
{
	LinearLayer *layer;
	LinearLayerEvaluation *eval;
	LinearLayerGrad *grad;
} LinearLayerSGD;

LinearLayerSGD *linlay_sgd_alloc(LinearLayer *linlay);

void linlay_sgd_free(LinearLayerSGD *linlay_sgd);

void linlay_sgd_forward(LinearLayerSGD *linlay_sgd, double *input);

void linlay_sgd_backward(LinearLayerSGD *linlay_sgd, double *input, double *grad_loss_out, double step);

typedef struct ConvolutionalLayerSGD
{
	ConvolutionalLayer *layer;
	ConvolutionalLayerEvaluation *eval;
	ConvolutionalLayerGrad* grad;
} ConvolutionalLayerSGD;

ConvolutionalLayerSGD* cnn_layer_sgd_alloc(ConvolutionalLayer* cnn_layer);

void cnn_layer_sgd_free(ConvolutionalLayerSGD* cnn_layer_sgd);

void cnn_layer_sgd_forward(ConvolutionalLayerSGD* cnn_layer_sgd, ImageLayer* input);

void cnn_layer_sgd_backward(ConvolutionalLayerSGD* cnn_layer_sgd, ImageLayer* input, ImageLayer* grad_loss_out, double step);
