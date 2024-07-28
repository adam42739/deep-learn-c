#include "activation.h"

void _activation_function(double* pre_activation, double* output, int len, __activation_type act_type)
{
	switch (act_type)
	{
	case ACT_IDENTITY:
		_activation_identity(pre_activation, output, len);
		break;
	case ACT_RELU:
		_activation_relu(pre_activation, output, len);
		break;
	case ACT_SOFTMAX:
		_activation_softmax(pre_activation, output, len);
		break;
	case ACT_HYPERTAN:
		_activation_hypertan(pre_activation, output, len);
		break;
	default:
		break;
	}
}

void _activation_identity(double* pre_activation, double* output, int len)
{
	for (int i = 0; i < len; ++i)
	{
		output[i] = pre_activation[i];
	}
}

void _activation_relu(double* pre_activation, double* output, int len)
{
	for (int i = 0; i < len; ++i)
	{
		output[i] = (pre_activation[i] > 0 ? pre_activation[i] : 0);
	}
}

void _activation_softmax(double* pre_activation, double* output, int len)
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

void _activation_hypertan(double* pre_activation, double* output, int len)
{
	for (int i = 0; i < len; ++i)
	{
		output[i] = tanh(pre_activation[i]);
	}
}

void _activation_deriv(double* pre_activation, double* output, double* deriv, int len, __activation_type act_type)
{
	switch (act_type)
	{
	case ACT_IDENTITY:
		_activation_identity_deriv(pre_activation, output, deriv, len);
		break;
	case ACT_RELU:
		_activation_relu_deriv(pre_activation, output, deriv, len);
		break;
	case ACT_SOFTMAX:
		_activation_softmax_deriv(pre_activation, output, deriv, len);
		break;
	case ACT_HYPERTAN:
		_activation_hypertan_deriv(pre_activation, output, deriv, len);
		break;
	default:
		break;
	}
}

void _activation_identity_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = 1;
	}
}

void _activation_relu_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = (pre_activation[i] > 0 ? 1 : 0);
	}
}

void _activation_softmax_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = output[i] * (1 - output[i]);
	}
}

void _activation_hypertan_deriv(double* pre_activation, double* output, double* deriv, int len)
{
	for (int i = 0; i < len; ++i)
	{
		deriv[i] = 1 - pow(output[i], 2);
	}
}
