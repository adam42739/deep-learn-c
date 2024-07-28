#include "convolutional_filter.h"

ConvolutionalFilter* cnn_filter_alloc(int m, int n, int num_indexes)
{
	ConvolutionalFilter* cnn_filter = _mem_alloc(sizeof(ConvolutionalFilter));
	cnn_filter->linmod = linmod_alloc(m * n);
	cnn_filter->m = m;
	cnn_filter->n = n;
	cnn_filter->img_array_indexes = _mem_alloc(sizeof(int) * num_indexes);
	cnn_filter->num_indexes = num_indexes;
	return cnn_filter;
}

void cnn_filter_free(ConvolutionalFilter* cnn_filter)
{
	linmod_free(cnn_filter->linmod);
	free(cnn_filter->img_array_indexes);
	free(cnn_filter);
}

void cnn_filter_randomize_weights(ConvolutionalFilter* cnn_filter, __rng_dist_type rng_type)
{
	linmod_randomize_weights(cnn_filter->linmod, rng_type);
}

void cnn_filter_set_bias_zero(ConvolutionalFilter* cnn_filter)
{
	linmod_set_bias_zero(cnn_filter->linmod);
}

double cnn_filter_forward_at_index(ConvolutionalFilter* cnn_filter, ImageLayer* img_layer, int i_i, int j_i)
{
	double sum_prod = 0;
	int img_pixel_start = img_layer->n * i_i + j_i;
	int filter_pixel = 0;
	for (int k = 0; k < cnn_filter->num_indexes; ++k)
	{
		int img_pixel = img_pixel_start;
		for (int di = 0; di < cnn_filter->m; ++di)
		{
			for (int dj = 0; dj < cnn_filter->n; ++dj)
			{
				sum_prod += cnn_filter->linmod->weights[filter_pixel] * img_layer->images[k]->pixels[img_pixel];
				++filter_pixel;
				++img_pixel;
			}
		}
	}
	return sum_prod + cnn_filter->linmod->bias;
}

void cnn_filter_forward_image(ConvolutionalFilter* cnn_filter, ImageLayer* img_layer)
{
	for (int i = 0; i < img_layer->m - cnn_filter->m + 1; ++i)
	{
		for (int j = 0; j < img_layer->n - cnn_filter->n + 1; ++j)
		{
			double output = cnn_filter_forward_at_index(cnn_filter, img_layer, i, j);
			// TODO
		}
	}
}

void cnn_filter_forward(ConvolutionalFilter* cnn_filter, ImageLayer* img_layer)
{

}
