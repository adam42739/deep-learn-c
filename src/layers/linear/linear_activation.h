#pragma once
#include <math.h>

typedef int __linear_activation_type;

#define LINEAR_ACT_IDENTITY 0
#define LINEAR_ACT_RELU 1
#define LINEAR_ACT_SOFTMAX 2
#define LINEAR_ACT_HYPERTAN 3

void _linear_activation_function(double* pre_activation, double* output, int len, __linear_activation_type act_type);

void _linear_activation_identity(double* pre_activation, double* output, int len);

void _linear_activation_relu(double* pre_activation, double* output, int len);

void _linear_activation_softmax(double* pre_activation, double* output, int len);

void _linear_activation_hypertan(double* pre_activation, double* output, int len);

void _linear_activation_deriv(double* pre_activation, double* output, double* deriv, int len, __linear_activation_type act_type);

void _linear_activation_identity_deriv(double* pre_activation, double* output, double* deriv, int len);

void _linear_activation_relu_deriv(double* pre_activation, double* output, double* deriv, int len);

void _linear_activation_softmax_deriv(double* pre_activation, double* output, double* deriv, int len);

void _linear_activation_hypertan_deriv(double* pre_activation, double* output, double* deriv, int len);
