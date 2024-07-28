#pragma once
#include <math.h>

typedef int __activation_type;

#define ACT_IDENTITY 0
#define ACT_RELU 1
#define ACT_SOFTMAX 2
#define ACT_HYPERTAN 3

void _activation_function(double* pre_activation, double* output, int len, __activation_type act_type);

void _activation_identity(double* pre_activation, double* output, int len);

void _activation_relu(double* pre_activation, double* output, int len);

void _activation_softmax(double* pre_activation, double* output, int len);

void _activation_hypertan(double* pre_activation, double* output, int len);

void _activation_deriv(double* pre_activation, double* output, double* deriv, int len, __activation_type act_type);

void _activation_identity_deriv(double* pre_activation, double* output, double* deriv, int len);

void _activation_relu_deriv(double* pre_activation, double* output, double* deriv, int len);

void _activation_softmax_deriv(double* pre_activation, double* output, double* deriv, int len);

void _activation_hypertan_deriv(double* pre_activation, double* output, double* deriv, int len);
