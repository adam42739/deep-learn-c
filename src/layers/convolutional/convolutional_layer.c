#include "convolutional_layer.h"

ConvolutionalLayer *cnn_layer_alloc(
	int num_input_img,
	int filter_m,
	int filter_n,
	int input_m,
	int input_n,
	int num_filters,
	int *num_filters_index,
	int **filter_index,
	__img_activation_type act_type)
{
	ConvolutionalLayer *cnn_layer = _mem_alloc(sizeof(ConvolutionalLayer));
	cnn_layer->num_input_img = num_input_img;
	cnn_layer->input_m = input_m;
	cnn_layer->input_n = input_n;
	cnn_layer->filters = _mem_alloc(sizeof(ConvolutionalMultiFilter *) * num_filters);
	for (int i = 0; i < num_filters; ++i)
	{
		cnn_layer->filters[i] = cnn_mfilter_alloc(num_filters_index[i], filter_index[i], filter_m, filter_n, input_m, input_n, act_type);
	}
	cnn_layer->num_filters = num_filters;
	cnn_layer->output_m = input_m - filter_m + 1;
	cnn_layer->output_n = input_n - filter_n + 1;
	cnn_layer->act_type = act_type;
	return cnn_layer;
}

void cnn_layer_free(ConvolutionalLayer *cnn_layer)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_mfilter_free(cnn_layer->filters[i]);
	}
	free(cnn_layer->filters);
	free(cnn_layer);
}

void cnn_layer_randomize_weights(ConvolutionalLayer *cnn_layer, double stdev)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_mfilter_randomize_weights(cnn_layer->filters[i], stdev);
	}
}

void cnn_layer_set_bias_zero(ConvolutionalLayer *cnn_layer)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_mfilter_set_bias_zero(cnn_layer->filters[i]);
	}
}

ConvolutionalLayerEvaluation *cnn_layer_eval_alloc(ConvolutionalLayer *cnn_layer)
{
	ConvolutionalLayerEvaluation *cnn_layer_eval = _mem_alloc(sizeof(ConvolutionalLayerEvaluation));
	cnn_layer_eval->output = img_layer_alloc(cnn_layer->num_filters, cnn_layer->output_m, cnn_layer->output_n);
	cnn_layer_eval->filter_evals = _mem_alloc(sizeof(ConvolutionalMultiFilterEvaluation *) * cnn_layer->num_filters);
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_layer_eval->filter_evals[i] = cnn_mfilter_eval_alloc(cnn_layer->filters[i], cnn_layer_eval->output->img_arrays[i]);
	}
	cnn_layer_eval->num_filters = cnn_layer->num_filters;
	return cnn_layer_eval;
}

void cnn_layer_eval_free(ConvolutionalLayerEvaluation *cnn_layer_eval)
{
	img_layer_free(cnn_layer_eval->output);
	for (int i = 0; i < cnn_layer_eval->num_filters; ++i)
	{
		cnn_mfilter_eval_free(cnn_layer_eval->filter_evals[i]);
	}
	free(cnn_layer_eval->filter_evals);
	free(cnn_layer_eval);
}

void cnn_layer_eval_compute(ConvolutionalLayer *cnn_layer, ImageLayer *input, ConvolutionalLayerEvaluation *cnn_layer_eval)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_mfilter_eval_compute(cnn_layer_eval->filter_evals[i], cnn_layer->filters[i], input);
	}
}

ConvolutionalLayerGrad *cnn_layer_grad_alloc(ConvolutionalLayer *cnn_layer)
{
	ConvolutionalLayerGrad *cnn_layer_grad = _mem_alloc(sizeof(ConvolutionalLayerGrad));
	cnn_layer_grad->input_m = cnn_layer->input_m;
	cnn_layer_grad->input_n = cnn_layer->input_n;
	cnn_layer_grad->output_m = cnn_layer->output_m;
	cnn_layer_grad->output_n = cnn_layer->output_n;
	cnn_layer_grad->num_input_img = cnn_layer->num_input_img;
	cnn_layer_grad->num_filters = cnn_layer->num_filters;
	cnn_layer_grad->filter_grads = _mem_alloc(sizeof(ConvolutionalMultiFilterGrad *) * cnn_layer->num_filters);
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		cnn_layer_grad->filter_grads[i] = cnn_mfilter_grad_alloc(cnn_layer->filters[i]);
	}
	cnn_layer_grad->grad_loss_input = img_layer_alloc(cnn_layer->num_input_img, cnn_layer->input_m, cnn_layer->input_n);
	return cnn_layer_grad;
}

void cnn_layer_grad_free(ConvolutionalLayerGrad *cnn_layer_grad)
{
	for (int i = 0; i < cnn_layer_grad->num_filters; ++i)
	{
		cnn_mfilter_grad_free(cnn_layer_grad->filter_grads[i]);
	}
	free(cnn_layer_grad->filter_grads);
	img_layer_free(cnn_layer_grad->grad_loss_input);
	free(cnn_layer_grad);
}

void cnn_layer_grad_compute_input(ConvolutionalLayerGrad *cnn_layer_grad, ConvolutionalLayer *cnn_layer)
{
	for (int i = 0; i < cnn_layer->num_filters; ++i)
	{
		
	}
}

void cnn_layer_grad_compute(
	ConvolutionalLayerGrad *cnn_layer_grad,
	ConvolutionalLayerEvaluation *cnn_layer_eval,
	ConvolutionalLayer *cnn_layer,
	ImageLayer *input,
	ImageLayer *grad_loss_out)
{
	for (int i = 0; i < cnn_layer_grad->num_filters; ++i)
	{
		cnn_mfilter_grad_compute(cnn_layer_grad->filter_grads[i], cnn_layer->filters[i], cnn_layer_eval->filter_evals[i], input, grad_loss_out->img_arrays[i]);
	}
	cnn_layer_grad_compute_input(cnn_layer_grad, cnn_layer);
}
