#pragma once
#include <math.h>

typedef int __loss_type;

#define LOSS_MSE 0

double _loss_function(double *output, double *expected, int n, __loss_type loss_type);

double _loss_mse(double *output, double *expected, int n);

void _loss_deriv(double *output, double *expected, double* deriv, int n, __loss_type loss_type);

void _loss_mse_deriv(double *output, double *expected, double* deriv, int n);
