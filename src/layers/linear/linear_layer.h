#pragma once
#include "linear_model.h"
#include "linear_activation.h"

typedef struct LinearLayer
{
	LinearModel **linmods;
	int num_linmods;
	__linear_activation_type act_type;
} LinearLayer;

LinearLayer *linlay_alloc(int input_size, int num_linmods, __linear_activation_type act_type);

void linlay_free(LinearLayer *linlay);

void linlay_randomize_weights(LinearLayer *linlay, double stdev);

void linlay_set_bias_zero(LinearLayer *linlay);

void linlay_forward(LinearLayer *linlay, double *input, double *pre_activation, double *output);

typedef struct LinearLayerEvaluation
{
	double *pre_activation;
	double *output;
} LinearLayerEvaluation;

LinearLayerEvaluation *linlay_eval_alloc(LinearLayer *linlay);

void linlay_eval_free(LinearLayerEvaluation *linlay_eval);

void linlay_eval_compute(LinearLayerEvaluation *linlay_eval, LinearLayer *linlay, double *input);

typedef struct LinearLayerGrad
{
	LinearModelGrad **linmod_grads;
	int num_linmod_grads;
	double *grad_out_net;
	double *grad_loss_net;
	double *grad_loss_input;
} LinearLayerGrad;

LinearLayerGrad *linlay_grad_alloc(LinearLayer *linlay);

void linlay_grad_free(LinearLayerGrad *linlay_grad);

void linlay_grad_compute_net(LinearLayerGrad *linlay_grad, LinearLayer *linlay, LinearLayerEvaluation *linlay_eval, double *grad_loss_output);

void linlay_grad_compute_linmod(LinearLayerGrad *linlay_grad, LinearLayer *linlay, double *input);

void linlay_grad_compute_input(LinearLayerGrad *linlay_grad, LinearLayer *linlay, double *grad_loss_output);

void linlay_grad_compute(LinearLayerGrad *linlay_grad, LinearLayer *linlay, LinearLayerEvaluation *linlay_eval, double *input, double *grad_loss_output);
