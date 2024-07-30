#include "vanilla_sgd.h"

LinearLayerSGD *linlay_sgd_alloc(LinearLayer *linlay)
{
	LinearLayerSGD *linlay_sgd = _mem_alloc(sizeof(LinearLayerSGD));
	linlay_sgd->layer = linlay;
	linlay_sgd->eval = linlay_eval_alloc(linlay);
	linlay_sgd->grad = linlay_grad_alloc(linlay);
	return linlay_sgd;
}

void linlay_sgd_free(LinearLayerSGD *linlay_sgd)
{
	linlay_eval_free(linlay_sgd->eval);
	linlay_grad_free(linlay_sgd->grad);
	free(linlay_sgd);
}

void linlay_sgd_forward(LinearLayerSGD *linlay_sgd, double *input)
{
	linlay_eval_compute(linlay_sgd->eval, linlay_sgd->layer, input);
}

void linlay_sgd_backward(LinearLayerSGD *linlay_sgd, double *input, double *grad_loss_out, double step)
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

ConvolutionalLayerSGD *cnnlay_sgd_alloc(ConvolutionalLayer *cnn_layer)
{
	ConvolutionalLayerSGD *cnnlay_sgd = _mem_alloc(sizeof(ConvolutionalLayerSGD));
	cnnlay_sgd->layer = cnn_layer;
	cnnlay_sgd->eval = cnn_layer_eval_alloc(cnn_layer);
	cnnlay_sgd->grad = cnn_layer_grad_alloc(cnn_layer);
	return cnnlay_sgd;
}

void cnnlay_sgd_free(ConvolutionalLayerSGD *cnnlay_sgd)
{
	cnn_layer_eval_free(cnnlay_sgd->eval);
	cnn_layer_grad_free(cnnlay_sgd->grad);
	free(cnnlay_sgd);
}

void cnnlay_sgd_forward(ConvolutionalLayerSGD *cnnlay_sgd, ImageLayer *input)
{
	cnn_layer_eval_compute(cnnlay_sgd->layer, input, cnnlay_sgd->eval);
}

void cnnlay_sgd_backward(ConvolutionalLayerSGD *cnnlay_sgd, ImageLayer *input, ImageLayer *grad_loss_out, double step)
{
}
