#include "convolutional_filter.h"

ConvolutionalFilter* cnn_filter_alloc(int filter_m, int filter_n, int num_indexes, int* indexes)
{
	ConvolutionalFilter* cnn_filter = _mem_alloc(sizeof(ConvolutionalFilter));
	cnn_filter->linmod = linmod_alloc(filter_m * filter_n * num_indexes);
	cnn_filter->filter_m = filter_m;
	cnn_filter->filter_n = filter_n;
	cnn_filter->img_array_indexes = _mem_alloc(sizeof(int) * num_indexes);
	memcpy(cnn_filter->img_array_indexes, indexes, sizeof(int) * num_indexes);
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

double cnn_filter_forward_at_index(ConvolutionalFilter* cnn_filter, ImageLayer* input, int i_i, int j_i)
{
	double sum_prod = 0;
	int filter_pixel = 0;
	for (int k = 0; k < cnn_filter->num_indexes; ++k)
	{
		int img_index = cnn_filter->img_array_indexes[k];
		for (int di = 0; di < cnn_filter->filter_m; ++di)
		{
			for (int dj = 0; dj < cnn_filter->filter_n; ++dj)
			{
				int img_pixel = (i_i + di) * input->n + j_i + dj;
				sum_prod += cnn_filter->linmod->weights[filter_pixel] * input->images[img_index]->pixels[img_pixel];
				++filter_pixel;
			}
		}
	}
	return sum_prod + cnn_filter->bias;
}

void cnn_filter_forward(ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* output)
{
	int output_index = 0;
	for (int i = 0; i < input->m - cnn_filter->filter_m + 1; ++i)
	{
		for (int j = 0; j < input->n - cnn_filter->filter_n + 1; ++j)
		{
			output->pixels[output_index] = cnn_filter_forward_at_index(cnn_filter, input, i, j);
			++output_index;
		}
	}
}

ConvolutionalFilterGrad* cnn_filter_grad_alloc(ConvolutionalFilter* cnn_filter, int output_m, int output_n)
{
	ConvolutionalFilterGrad* cnn_filter_grad = _mem_alloc(sizeof(ConvolutionalFilterGrad));
	int num_grad = output_m * output_n;
	cnn_filter_grad->linmod_grads = _mem_alloc(sizeof(LinearModelGrad*) * num_grad);
	for (int i = 0; i < num_grad; ++i)
	{
		cnn_filter_grad->linmod_grads[i] = linmod_grad_alloc(cnn_filter->linmod);
	}
	cnn_filter_grad->output_m = output_m;
	cnn_filter_grad->output_n = output_n;
	return cnn_filter_grad;
}

void cnn_filter_grad_free(ConvolutionalFilterGrad* cnn_filter_grad)
{
	int num_grad = cnn_filter_grad->output_m * cnn_filter_grad->output_n;
	for (int i = 0; i < num_grad; ++i)
	{
		linmod_grad_free(cnn_filter_grad->linmod_grads[i]);
	}
	free(cnn_filter_grad->linmod_grads);
	free(cnn_filter_grad);
}

void cnn_filter_grad_compute_weights(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net)
{
	int output_index = 0;
	for (int oi = 0; oi < grad_loss_net->m; ++oi)
	{
		for (int oj = 0; oj < grad_loss_net->n; ++oj)
		{
			int weight_index = 0;
			for (int in = 0; in < cnn_filter->num_indexes; ++in)
			{
				int img_index = cnn_filter->img_array_indexes[in];
				for (int fi = 0; fi < cnn_filter->filter_m; ++fi)
				{
					for (int fj = 0; fj < cnn_filter->filter_n; ++fj)
					{
						int input_index = input->n * (oi + fi) + (oj + fj);
						cnn_filter_grad->linmod_grads[output_index]->grad_net_weights[weight_index] = input->images[img_index]->pixels[input_index];
						cnn_filter_grad->linmod_grads[output_index]->grad_loss_weights[weight_index] =
							cnn_filter_grad->linmod_grads[output_index]->grad_net_weights[weight_index] * grad_loss_net->pixels[output_index];
						++weight_index;
					}
				}
			}
			++output_index;
		}
	}
}

void cnn_filter_grad_compute_bias(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net)
{
	int output_index = 0;
	for (int oi = 0; oi < grad_loss_net->m; ++oi)
	{
		for (int oj = 0; oj < grad_loss_net->n; ++oj)
		{
			cnn_filter_grad->linmod_grads[output_index]->grad_net_bias = 1;
			cnn_filter_grad->linmod_grads[output_index]->grad_loss_bias =
				cnn_filter_grad->linmod_grads[output_index]->grad_net_bias * grad_loss_net->pixels[output_index];
			++output_index;
		}
	}
}

void cnn_filter_grad_compute_input(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net)
{
	int output_index = 0;
	for (int oi = 0; oi < grad_loss_net->m; ++oi)
	{
		for (int oj = 0; oj < grad_loss_net->n; ++oj)
		{
			int weight_index = 0;
			for (int in = 0; in < cnn_filter->num_indexes; ++in)
			{
				int img_index = cnn_filter->img_array_indexes[in];
				for (int fi = 0; fi < cnn_filter->filter_m; ++fi)
				{
					for (int fj = 0; fj < cnn_filter->filter_n; ++fj)
					{
						int input_index = input->n * (oi + fi) + (oj + fj);
						cnn_filter_grad->linmod_grads[output_index]->grad_net_input[weight_index] = cnn_filter->linmod->weights[weight_index];
						cnn_filter_grad->linmod_grads[output_index]->grad_loss_input[weight_index] =
							cnn_filter_grad->linmod_grads[output_index]->grad_net_input[weight_index] * grad_loss_net->pixels[output_index];
						++weight_index;
					}
				}
			}
			++output_index;
		}
	}
}

void cnn_filter_grad_compute(ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* grad_loss_net)
{
	cnn_filter_grad_compute_weights(cnn_filter_grad, cnn_filter, input, grad_loss_net);
	cnn_filter_grad_compute_bias(cnn_filter_grad, cnn_filter, input, grad_loss_net);
	cnn_filter_grad_compute_input(cnn_filter_grad, cnn_filter, input, grad_loss_net);
}
