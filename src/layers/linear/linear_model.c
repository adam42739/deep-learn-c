#include "linear_model.h"

LinearModel* linmod_alloc(int num_weights)
{
	LinearModel* linmod = _mem_alloc(sizeof(LinearModel));
	linmod->weights = _mem_alloc(sizeof(double) * num_weights);
	linmod->num_weights = num_weights;
	return linmod;
}

void linmod_free(LinearModel* linmod)
{
	free(linmod->weights);
	free(linmod);
}

void linmod_randomize_weights(LinearModel* linmod, double stdev)
{
	for (int i = 0; i < linmod->num_weights; ++i)
	{
		linmod->weights[i] = _rng_normal(0, stdev);
	}
}

void linmod_set_bias_zero(LinearModel* linmod)
{
	linmod->bias = 0;
}

double linmod_forward(LinearModel* linmod, double* input)
{
	double value = linmod->bias;
	for (int i = 0; i < linmod->num_weights; ++i)
	{
		value += linmod->weights[i] * input[i];
	}
	return value;
}


LinearModelGrad* linmod_grad_alloc(LinearModel* linmod)
{
	LinearModelGrad* linmod_grad = _mem_alloc(sizeof(LinearModelGrad));
	linmod_grad->grad_net_weights = _mem_alloc(sizeof(double) * linmod->num_weights);
	linmod_grad->grad_loss_weights = _mem_alloc(sizeof(double) * linmod->num_weights);
	linmod_grad->grad_net_input = _mem_alloc(sizeof(double) * linmod->num_weights);
	linmod_grad->grad_loss_input = _mem_alloc(sizeof(double) * linmod->num_weights);
	return linmod_grad;
}

void linmod_grad_free(LinearModelGrad* linmod_grad)
{
	free(linmod_grad->grad_net_weights);
	free(linmod_grad->grad_loss_weights);
	free(linmod_grad->grad_net_input);
	free(linmod_grad->grad_loss_input);
	free(linmod_grad);
}

void linmod_grad_compute_weights(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net)
{
	for (int i = 0; i < linmod->num_weights; ++i)
	{
		linmod_grad->grad_net_weights[i] = input[i];
		linmod_grad->grad_loss_weights[i] = linmod_grad->grad_net_weights[i] * grad_loss_net;
	}
}

void linmod_grad_compute_bias(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net)
{
	linmod_grad->grad_net_bias = 1;
	linmod_grad->grad_loss_bias = linmod_grad->grad_net_bias * grad_loss_net;
}

void linmod_grad_compute_input(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net)
{
	for (int i = 0; i < linmod->num_weights; ++i)
	{
		linmod_grad->grad_net_input[i] = linmod->weights[i];
		linmod_grad->grad_loss_input[i] = linmod_grad->grad_net_input[i] * grad_loss_net;
	}
}

void linmod_grad_compute(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net)
{
	linmod_grad_compute_weights(linmod_grad, linmod, input, grad_loss_net);
	linmod_grad_compute_bias(linmod_grad, linmod, input, grad_loss_net);
	linmod_grad_compute_input(linmod_grad, linmod, input, grad_loss_net);
}
