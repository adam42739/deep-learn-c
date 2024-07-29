#include "linear_activation.h"

void _linear_activation_function(double* pre_activation, double* output, int len, __linear_activation_type act_type)
{
	switch (act_type)
	{
	case LINEAR_ACT_IDENTITY:
		_linear_activation_identity(pre_activation, output, len);
		break;
	case LINEAR_ACT_RELU:
		_linear_activation_relu(pre_activation, output, len);
		break;
	case LINEAR_ACT_SOFTMAX:
		_linear_activation_softmax(pre_activation, output, len);
		break;
	case LINEAR_ACT_HYPERTAN:
		_linear_activation_hypertan(pre_activation, output, len);
		break;
	default:
		break;
	}
}

void _linear_activation_identity(double* pre_activation, double* output, int len)
{
	for (int i = 0; i < len; ++i)
	{
		output[i] = pre_activation[i];
	}
}

void _linear_activation_relu(double* pre_activation, double* output, int len)
{
	for (int i = 0; i < len; ++i)
	{
		output[i] = (pre_activation[i] > 0 ? pre_activation[i] : 0);
	}
}

void _linear_activation_softmax(double* pre_activation, double* output, int len)
{
	double total = 0;
	for (int i = 0; i < len; ++i)
	{
		output[i] = exp(pre_activation[i]);
		total += output[i];
	}
	for (int i = 0; i < len; ++i)
	{
		output[i] /= total;
	}
}

void _linear_activation_hypertan(double* pre_activation, double* output, int len)
{
	for (int i = 0; i < len; ++i)
	{
		output[i] = tanh(pre_activation[i]);
	}
}

void _linear_activation_deriv(double* pre_activation, double* output, double* deriv, int len, __linear_activation_type act_type)
{
	switch (act_type)
	{
	case LINEAR_ACT_IDENTITY:
		_linear_activation_identity_deriv(pre_activation, output, deriv, len);
		break;
	case LINEAR_ACT_RELU:
		_linear_activation_relu_deriv(pre_activation, output, deriv, len);
		break;
	case LINEAR_ACT_SOFTMAX:
		_linear_activation_softmax_deriv(pre_activation, output, deriv, len);
		break;
	case LINEAR_ACT_HYPERTAN:
		_linear_activation_hypertan_deriv(pre_activation, output, deriv, len);
		break;
	default:
		break;
	}
}

void _linear_activation_identity_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = 1;
	}
}

void _linear_activation_relu_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = (pre_activation[i] > 0 ? 1 : 0);
	}
}

void _linear_activation_softmax_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = output[i] * (1 - output[i]);
	}
}

void _linear_activation_hypertan_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = 1 - pow(output[i], 2);
	}
}
