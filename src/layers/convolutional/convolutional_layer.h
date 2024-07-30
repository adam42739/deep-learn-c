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

ConvolutionalLayer cnn_layer_alloc(int num_input_img, int input_m, int input_n, int output_m, int output_n, int num_filter, int* filter_index);

void cnn_layer_free(ConvolutionalLayer* cnn_layer);

void cnn_layer_randomize_weights(ConvolutionalLayer* cnn_layer, double stdev);

void cnn_layer_set_bias_zero(ConvolutionalLayer* cnn_layer);
