#include "convolutional_filter.h"

typedef struct ConvolutionalLayer
{
	int num_input_img;
	int input_m;
	int input_n;
	ConvolutionalMultiFilter** filters;
	int num_filters;
	int output_m;
	int output_n;
	__img_activation_type act_type;
} ConvolutionalLayer;

ConvolutionalLayer* cnn_layer_alloc(
	int num_input_img,
	int filter_m,
	int filter_n,
	int input_m,
	int input_n,
	int num_filters,
	int* num_filters_index,
	int** filter_index,
	__img_activation_type act_type);

void cnn_layer_free(ConvolutionalLayer* cnn_layer);

void cnn_layer_randomize_weights(ConvolutionalLayer* cnn_layer, double stdev);

void cnn_layer_set_bias_zero(ConvolutionalLayer* cnn_layer);

typedef struct ConvolutionLayerEvaluation
{
	ImageLayer* output;
	ConvolutionalMultiFilterEvaluation** filter_evals;
	int num_filters;
} ConvolutionalLayerEvaluation;

ConvolutionalLayerEvaluation* cnn_layer_eval_alloc(ConvolutionalLayer* cnn_layer);

void cnn_layer_eval_free(ConvolutionalLayerEvaluation* cnn_layer_eval);

void cnn_layer_eval_compute(ConvolutionalLayer* cnn_layer, ImageLayer* input, ConvolutionalLayerEvaluation* cnn_layer_eval);

typedef struct ConvolutionalLayerGrad
{
	int input_m;
	int input_n;
	int output_m;
	int output_n;
	int num_input_img;
	int num_filters;
	ConvolutionalMultiFilterGrad** filter_grads;
} ConvolutionalLayerGrad;

ConvolutionalLayerGrad* cnn_layer_grad_alloc(ConvolutionalLayer* cnn_layer);

void cnn_layer_grad_free(ConvolutionalLayerGrad* cnn_layer_grad);

void cnn_layer_grad_compute(
	ConvolutionalLayerGrad* cnn_layer_grad,
	ConvolutionalLayerEvaluation* cnn_layer_eval,
	ConvolutionalLayer* cnn_layer,
	ImageLayer* input,
	ImageLayer* grad_loss_out);
