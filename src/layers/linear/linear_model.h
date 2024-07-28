#pragma once
#include "../../general_lib/random.h"
#include "../../general_lib/stdlib_mod.h"

typedef struct LinearModel
{
	double* weights;
	int num_weights;
	double bias;
} LinearModel;

LinearModel* linmod_alloc(int num_weights);

void linmod_free(LinearModel* linmod);

void linmod_randomize_weights(LinearModel* linmod, __rng_dist_type rng_type);

void linmod_set_bias_zero(LinearModel* linmod);

double linmod_forward(LinearModel* linmod, double* input);

typedef struct LinearModelGrad
{
	double* grad_net_weights;
	double* grad_loss_weights;
	double grad_net_bias;
	double grad_loss_bias;
	double* grad_net_input;
	double* grad_loss_input;
} LinearModelGrad;

LinearModelGrad* linmod_grad_alloc(LinearModel* linmod);

void linmod_grad_free(LinearModelGrad* linmod_grad);

void linmod_grad_compute_weights(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net);

void linmod_grad_compute_bias(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net);

void linmod_grad_compute_input(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net);

void linmod_grad_compute(LinearModelGrad* linmod_grad, LinearModel* linmod, double* input, double grad_loss_net);
