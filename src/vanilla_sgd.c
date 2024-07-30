#include "vanilla_sgd.h"

LinearLayerSGD* linlay_sgd_alloc(LinearLayer* linlay)
{
	LinearLayerSGD* linlay_sgd = _mem_alloc(sizeof(LinearLayerSGD));
	linlay_sgd->layer = linlay;
	linlay_sgd->eval = linlay_eval_alloc(linlay);
	linlay_sgd->grad = linlay_grad_alloc(linlay);
	return linlay_sgd;
}

void linlay_sgd_free(LinearLayerSGD* linlay_sgd)
{
	linlay_eval_free(linlay_sgd->eval);
	linlay_grad_free(linlay_sgd->grad);
	free(linlay_sgd);
}

void linlay_sgd_forward(LinearLayerSGD* linlay_sgd, double* input)
{
	linlay_eval_compute(linlay_sgd->eval, linlay_sgd->layer, input);
}

void linlay_sgd_backward(LinearLayerSGD* linlay_sgd, double* input, double* grad_loss_out, double step)
{
	linlay_grad_compute(linlay_sgd->grad, linlay_sgd->layer, linlay_sgd->eval, input, grad_loss_out);
	for (int i = 0; i < linlay_sgd->layer->num_linmods; ++i)
	{
		linlay_sgd->layer->linmods[i]->bias -= step * linlay_sgd->grad->linmod_grads[i]->grad_loss_bias;
	}
	for (int i = 0; i < linlay_sgd->layer->num_linmods; ++i)
	{
		for (int j = 0; j < linlay_sgd->layer->linmods[i]->num_weights; ++j)
		{
			linlay_sgd->layer->linmods[i]->weights[j] -= step * linlay_sgd->grad->linmod_grads[i]->grad_loss_weights[j];
		}
	}
}

ConvolutionalLayerSGD* cnn_layer_sgd_alloc(ConvolutionalLayer* cnn_layer)
{
	ConvolutionalLayerSGD* cnn_layer_sgd = _mem_alloc(sizeof(ConvolutionalLayerSGD));
	cnn_layer_sgd->layer = cnn_layer;
	cnn_layer_sgd->eval = cnn_layer_eval_alloc(cnn_layer);
	cnn_layer_sgd->grad = cnn_layer_grad_alloc(cnn_layer);
	return cnn_layer_sgd;
}

void cnn_layer_sgd_free(ConvolutionalLayerSGD* cnn_layer_sgd)
{
	cnn_layer_eval_free(cnn_layer_sgd->eval);
	cnn_layer_grad_free(cnn_layer_sgd->grad);
	free(cnn_layer_sgd);
}

void cnn_layer_sgd_forward(ConvolutionalLayerSGD* cnnlay_sgd, ImageLayer* input)
{
	cnn_layer_eval_compute(cnnlay_sgd->layer, input, cnnlay_sgd->eval);
}

void cnn_layer_sgd_backward_output_index(
	ConvolutionalLayerSGD* cnn_layer_sgd,
	double step,
	int filter_num,
	int prenet_num,
	int output_i,
	int ouput_j)
{
	int filter_m = cnn_layer_sgd->grad->filter_grads[filter_num]->filter_grads[prenet_num]->filter_m;
	int filter_n = cnn_layer_sgd->grad->filter_grads[filter_num]->filter_grads[prenet_num]->filter_n;
	double** grad_loss_weight = cnn_layer_sgd->grad->filter_grads[filter_num]->filter_grads[prenet_num]->filteri_grads[output_i][output_i]->grad_loss_weights;
	double** weights = cnn_layer_sgd->layer->filters[filter_num]->filters[prenet_num]->weights;
	for (int i = 0; i < filter_m; ++i)
	{
		for (int j = 0; j < filter_n; ++j)
		{
			weights[i][j] -= step * grad_loss_weight[i][j];
		}
	}
}

void cnn_layer_sgd_backward_prenet(ConvolutionalLayerSGD* cnn_layer_sgd, double step, int filter_num, int prenet_num)
{
	ImageArray* grad_loss_bias = cnn_layer_sgd->grad->filter_grads[filter_num]->grad_loss_bias;
	for (int i = 0; i < cnn_layer_sgd->grad->output_m; ++i)
	{
		for (int j = 0; j < cnn_layer_sgd->grad->output_n; ++j)
		{
			cnn_layer_sgd_backward_output_index(cnn_layer_sgd, step, filter_num, prenet_num, i, j);
			cnn_layer_sgd->layer->filters[filter_num]->bias -= step * grad_loss_bias->pixels[i][j];
		}
	}
}

void cnn_layer_sgd_backward_filter(ConvolutionalLayerSGD* cnn_layer_sgd, double step, int filter_num)
{
	for (int i = 0; i < cnn_layer_sgd->grad->filter_grads[filter_num]->num_filters; ++i)
	{
		cnn_layer_sgd_backward_prenet(cnn_layer_sgd, step, filter_num, i);
	}
}

void cnn_layer_sgd_backward(ConvolutionalLayerSGD* cnn_layer_sgd, ImageLayer* input, ImageLayer* grad_loss_out, double step)
{
	cnn_layer_grad_compute(cnn_layer_sgd->grad, cnn_layer_sgd->eval, cnn_layer_sgd->layer, input, grad_loss_out);
	for (int i = 0; i < cnn_layer_sgd->layer->num_filters; ++i)
	{
		cnn_layer_sgd_backward_filter(cnn_layer_sgd, step, i);
	}
}
