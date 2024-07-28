#pragma once
#include "linear_network.h"

typedef struct LinearLayerSGD
{
	LinearLayer *layer;
	LinearLayerEvaluation *eval;
	LinearLayerGrad *grad;
} LinearLayerSGD;

LinearLayerSGD *linlay_sgd_alloc(LinearLayer *linlay);

void linlay_sgd_free(LinearLayerSGD *linlay_sgd);

void linlay_sgd_forward(LinearLayerSGD *linlay_sgd, double *input);

void linlay_sgd_backward(LinearLayerSGD *linlay_sgd, double *input, double *grad_loss_out, double step);
