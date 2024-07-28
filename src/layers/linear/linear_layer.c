#include "linear_layer.h"

LinearLayer *linlay_alloc(int input_size, int num_linmods, __activation_type act_type)
{
	LinearLayer *linlay = _mem_alloc(sizeof(LinearLayer));
	linlay->linmods = _mem_alloc(sizeof(LinearModel *) * num_linmods);
	for (int i = 0; i < num_linmods; ++i)
	{
		linlay->linmods[i] = linmod_alloc(input_size);
	}
	linlay->num_linmods = num_linmods;
	linlay->act_type = act_type;
	return linlay;
}

void linlay_free(LinearLayer *linlay)
{
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		linmod_free(linlay->linmods[i]);
	}
	free(linlay->linmods);
	free(linlay);
}

void linlay_randomize_weights(LinearLayer *linlay, __activation_type rng_type)
{
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		linmod_randomize_weights(linlay->linmods[i], rng_type);
	}
}

void linlay_set_bias_zero(LinearLayer *linlay)
{
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		linmod_set_bias_zero(linlay->linmods[i]);
	}
}

void linlay_forward(LinearLayer *linlay, double *input, double *pre_activation, double *output)
{
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		pre_activation[i] = linmod_forward(linlay->linmods[i], input);
	}
	_activation_function(pre_activation, output, linlay->num_linmods, linlay->act_type);
}

LinearLayerEvaluation *linlay_eval_alloc(LinearLayer *linlay)
{
	LinearLayerEvaluation *linlay_eval = _mem_alloc(sizeof(LinearLayerEvaluation));
	linlay_eval->pre_activation = _mem_alloc(sizeof(double) * linlay->num_linmods);
	linlay_eval->output = _mem_alloc(sizeof(double) * linlay->num_linmods);
	return linlay_eval;
}

void linlay_eval_free(LinearLayerEvaluation *linlay_eval)
{
	free(linlay_eval->pre_activation);
	free(linlay_eval->output);
	free(linlay_eval);
}

void linlay_eval_compute(LinearLayerEvaluation *linlay_eval, LinearLayer *linlay, double *input)
{
	linlay_forward(linlay, input, linlay_eval->pre_activation, linlay_eval->output);
}

LinearLayerGrad *linlay_grad_alloc(LinearLayer *linlay)
{
	LinearLayerGrad *linlay_grad = _mem_alloc(sizeof(LinearLayerGrad));
	linlay_grad->linmod_grads = _mem_alloc(sizeof(LinearModelGrad *) * linlay->num_linmods);
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		linlay_grad->linmod_grads[i] = linmod_grad_alloc(linlay->linmods[i]);
	}
	linlay_grad->grad_out_net = _mem_alloc(sizeof(double) * linlay->num_linmods);
	linlay_grad->grad_loss_net = _mem_alloc(sizeof(double) * linlay->num_linmods);
	int num_inputs = linlay->linmods[0]->num_weights;
	linlay_grad->grad_loss_input = _mem_alloc(sizeof(double) * num_inputs);
	linlay_grad->num_linmod_grads = linlay->num_linmods;
	return linlay_grad;
}

void linlay_grad_free(LinearLayerGrad *linlay_grad)
{
	for (int i = 0; i < linlay_grad->num_linmod_grads; ++i)
	{
		linmod_grad_free(linlay_grad->linmod_grads[i]);
	}
	free(linlay_grad->linmod_grads);
	free(linlay_grad->grad_out_net);
	free(linlay_grad->grad_loss_net);
	free(linlay_grad->grad_loss_input);
	free(linlay_grad);
}

void linlay_grad_compute_net(LinearLayerGrad *linlay_grad, LinearLayer *linlay, LinearLayerEvaluation *linlay_eval, double *grad_loss_output)
{
	_activation_deriv(linlay_eval->pre_activation, linlay_eval->output, linlay_grad->grad_out_net, linlay->num_linmods, linlay->act_type);
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		linlay_grad->grad_loss_net[i] = grad_loss_output[i] * linlay_grad->grad_out_net[i];
	}
}

void linlay_grad_compute_linmod(LinearLayerGrad *linlay_grad, LinearLayer *linlay, double *input)
{
	for (int i = 0; i < linlay->num_linmods; ++i)
	{
		linmod_grad_compute(linlay_grad->linmod_grads[i], linlay->linmods[i], input, linlay_grad->grad_loss_net[i]);
	}
}

void linlay_grad_compute_input(LinearLayerGrad *linlay_grad, LinearLayer *linlay, double *grad_loss_output)
{
	for (int i = 0; i < linlay->linmods[0]->num_weights; ++i)
	{
		grad_loss_output[i] = 0;
		for (int j = 0; j < linlay->num_linmods; ++j)
		{
			grad_loss_output[i] += linlay_grad->linmod_grads[j]->grad_loss_input[i];
		}
	}
}

void linlay_grad_compute(LinearLayerGrad *linlay_grad, LinearLayer *linlay, LinearLayerEvaluation *linlay_eval, double *input, double *grad_loss_output)
{
	linlay_grad_compute_net(linlay_grad, linlay, linlay_eval, grad_loss_output);
	linlay_grad_compute_linmod(linlay_grad, linlay, input);
	linlay_grad_compute_input(linlay_grad, linlay, grad_loss_output);
}
