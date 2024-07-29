#include "convolutional_layer.h"

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
)
{
	ConvolutionalLayer* cnn_layer = _mem_alloc(sizeof(ConvolutionalLayer));
	cnn_layer->num_input_img = num_input_img;
	cnn_layer->input_m = input_m;
	cnn_layer->input_n = input_n;
	cnn_layer->filters = _mem_alloc(sizeof(ConvolutionalFilter*) * num_filters);
	for (int i = 0; i < num_filters; ++i)
	{
		cnn_layer->filters[i] = cnn_filter_alloc(filter_m, filter_n, num_indexes[i], indexes[i]);
	}
	cnn_layer->num_filters = num_filters;
	cnn_layer->output_m = input_m - filter_m + 1;
	cnn_layer->output_n = input_n - filter_n + 1;
	cnn_layer->act_type = act_type;
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

void cnn_layer_forward(ConvolutionalLayer* cnn_layer, ImageLayer* input, ImageLayer* pre_activation, ImageLayer* output)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_filter_forward(cnn_layer->filters[i], input, pre_activation->images[i]);
		_activation_function(pre_activation->images[i]->pixels, output->images[i]->pixels, pre_activation->m * pre_activation->n, cnn_layer->act_type);
	}
}

ConvolutionalLayerEvaluation* cnn_layer_eval_alloc(ConvolutionalLayer* cnn_layer)
{
	ConvolutionalLayerEvaluation* cnn_layer_eval = _mem_alloc(sizeof(ConvolutionalLayerEvaluation));
	cnn_layer_eval->pre_activation = img_layer_alloc(cnn_layer->num_input_img, cnn_layer->input_m, cnn_layer->input_n);
	cnn_layer_eval->output = img_layer_alloc(cnn_layer->num_input_img, cnn_layer->input_m, cnn_layer->input_n);
	return cnn_layer_eval;
}

void cnn_layer_eval_free(ConvolutionalLayerEvaluation* cnn_layer_eval)
{
	img_layer_free(cnn_layer_eval->pre_activation);
	img_layer_free(cnn_layer_eval->output);
	free(cnn_layer_eval);
}

void cnn_layer_eval_compute(ConvolutionalLayerEvaluation* cnn_layer_eval, ConvolutionalLayer* cnn_layer, ImageLayer* input)
{
	cnn_layer_forward(cnn_layer, input, cnn_layer_eval->pre_activation, cnn_layer_eval->output);
}

ConvolutionalLayerGrad* cnn_layer_grad_alloc(ConvolutionalLayer* cnn_layer)
{
	ConvolutionalLayerGrad* cnn_layer_grad = _mem_alloc(sizeof(ConvolutionalLayerGrad));
	cnn_layer_grad->cnn_filter_grads = _mem_alloc(sizeof(ConvolutionalFilterGrad*) * cnn_layer->num_filters);
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_layer_grad->cnn_filter_grads[i] = cnn_filter_grad_alloc(cnn_layer->filters[i], cnn_layer->output_m, cnn_layer->output_n);
	}
	cnn_layer_grad->grad_loss_net = img_layer_alloc(cnn_layer->num_filters, cnn_layer->output_m, cnn_layer->output_n);
	cnn_layer_grad->grad_out_net = img_layer_alloc(cnn_layer->num_filters, cnn_layer->output_m, cnn_layer->output_n);
	cnn_layer_grad->grad_loss_input = img_layer_alloc(cnn_layer->num_input_img, cnn_layer->input_m, cnn_layer->input_n);
	cnn_layer_grad->num_filters = cnn_layer->num_filters;
	return cnn_layer_grad;
}

void cnn_layer_grad_free(ConvolutionalLayerGrad* cnn_layer_grad)
{
	for (int i = 0; i < cnn_layer_grad->num_filters; ++i)
	{
		cnn_filter_grad_free(cnn_layer_grad->cnn_filter_grads[i]);
	}
	free(cnn_layer_grad->cnn_filter_grads);
	img_layer_free(cnn_layer_grad->grad_loss_net);
	img_layer_free(cnn_layer_grad->grad_out_net);
	free(cnn_layer_grad);
}


void cnn_layer_grad_compute_net(
	ConvolutionalLayerGrad* cnn_layer_grad,
	ConvolutionalLayer* cnn_layer,
	ConvolutionalLayerEvaluation* cnn_layer_eval,
	ImageLayer* grad_loss_out
)
{
	for (int i = 0; i < cnn_layer_grad->num_filters; ++i)
	{
		int num_pixels = cnn_layer->output_m * cnn_layer->output_n;
		_activation_deriv(
			cnn_layer_eval->pre_activation->images[i]->pixels,
			cnn_layer_eval->output->images[i]->pixels,
			cnn_layer_grad->grad_out_net->images[i]->pixels,
			num_pixels,
			cnn_layer->act_type
		);
		for (int j = 0; j < num_pixels; ++j)
		{
			cnn_layer_grad->grad_loss_net->images[i]->pixels[j] = grad_loss_out->images[i]->pixels[j] * cnn_layer_grad->grad_out_net->images[i]->pixels[j];
		}
	}
}

void cnn_layer_grad_compute_filters(ConvolutionalLayerGrad* cnn_layer_grad, ConvolutionalLayer* cnn_layer, ImageLayer* input)
{
	for (int i = 0; i < cnn_layer_grad->num_filters; ++i)
	{
		cnn_filter_grad_compute(cnn_layer_grad->cnn_filter_grads[i], cnn_layer->filters[i], input, cnn_layer_grad->grad_loss_net->images[i]);
	}
}

void cnn_layer_grad_compute_input(ConvolutionalLayerGrad* cnn_layer_grad, ConvolutionalLayer* cnn_layer, ImageLayer* grad_loss_out)
{
	for (int fn = 0; fn < cnn_layer_grad->num_filters; ++fn)
	{
		int output_pixel = 0;
		for (int oi = 0; oi < cnn_layer->output_m; ++oi)
		{
			for (int oj = 0; oj < cnn_layer->output_n; ++oj)
			{
				int input_pixel = 0;
				for (int findex = 0; findex < cnn_layer->filters[fn]->num_indexes; ++findex)
				{
					int img_index = cnn_layer->filters[fn]->img_array_indexes[findex];
					for (int ii = 0; ii < cnn_layer->filters[fn]->filter_m; ++ii)
					{
						for (int ij = 0; ij < cnn_layer->filters[fn]->filter_n; ++ij)
						{
							double grad = cnn_layer_grad->cnn_filter_grads[fn]->linmod_grads[output_pixel]->grad_loss_input[input_pixel];
							int input_image_pixel = (oi + ii) * cnn_layer->output_n + (oj + ij);
							cnn_layer_grad->grad_loss_input->images[img_index]->pixels[input_image_pixel] += grad;
							++input_pixel;
						}
					}
				}
				++output_pixel;
			}
		}
	}
}

void cnn_layer_grad_compute(
	ConvolutionalLayerGrad* cnn_layer_grad,
	ConvolutionalLayer* cnn_layer,
	ConvolutionalLayerEvaluation* cnn_layer_eval,
	ImageLayer* input,
	ImageLayer* grad_loss_out
)
{
	cnn_layer_grad_compute_net(cnn_layer_grad, cnn_layer, cnn_layer_eval, grad_loss_out);
	cnn_layer_grad_compute_filters(cnn_layer_grad, cnn_layer, input);
	cnn_layer_grad_compute_input(cnn_layer_grad, cnn_layer, grad_loss_out);
}
