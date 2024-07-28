#include "convolutional_layer.h"

ConvolutionalLayer* cnn_layer_alloc(int num_filters, int m, int n, int** indexes, int* num_indexes)
{
	ConvolutionalLayer* cnn_layer = _mem_alloc(sizeof(ConvolutionalLayer));
	cnn_layer->filters = _mem_alloc(sizeof(ConvolutionalFilter*) * num_filters);
	for (int i = 0; i < num_filters; ++i)
	{
		cnn_layer->filters[i] = cnn_filter_alloc(m, n, num_indexes[i], indexes[i]);
	}
	cnn_layer->num_filters = num_filters;
	return cnn_layer;
}

void cnn_layer_free(ConvolutionalLayer* cnn_layer)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_filter_free(cnn_layer->filters[i]);
	}
	free(cnn_layer->filters);
	free(cnn_layer);
}

void cnn_layer_randomize_weights(ConvolutionalLayer* cnn_layer, __rng_dist_type rng_dist)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_filter_randomize_weights(cnn_layer->filters[i], rng_dist);
	}
}

void cnn_layer_set_bias_zero(ConvolutionalLayer* cnn_layer)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_filter_set_bias_zero(cnn_layer->filters[i]);
	}
}

void cnn_layer_forward(ConvolutionalLayer* cnn_layer, ImageLayer* input, ImageLayer* output)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_filter_forward(cnn_layer->filters[i], input, output->images[i]);
	}
}
