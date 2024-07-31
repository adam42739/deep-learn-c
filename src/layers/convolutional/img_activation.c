#include "img_activation.h"

void _img_activation(ImageArray* pre_activation, ImageArray* output, __img_activation_type act_type)
{
	switch (act_type)
	{
	case IMG_ACT_IDENTITY:
		_img_activation_identity(pre_activation, output);
		break;
	case IMG_ACT_RELU:
		_img_activation_relu(pre_activation, output);
		break;
	default:
		break;
	}
}

void _img_activation_identity(ImageArray* pre_activation, ImageArray* output)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			output->pixels[i][j] = pre_activation->pixels[i][j];
		}
	}
}

void _img_activation_relu(ImageArray* pre_activation, ImageArray* output)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			output->pixels[i][j] = (pre_activation->pixels[i][j] > 0 ? pre_activation->pixels[i][j] : 0);
		}
	}
}

void _img_activation_deriv(ImageArray* pre_activation, ImageArray* output, ImageArray* deriv, __img_activation_type act_type)
{
	switch (act_type)
	{
	case IMG_ACT_IDENTITY:
		_img_activation_identity_deriv(pre_activation, output, deriv);
		break;
	case IMG_ACT_RELU:
		_img_activation_relu_deriv(pre_activation, output, deriv);
		break;
	default:
		break;
	}
}

void _img_activation_identity_deriv(ImageArray* pre_activation, ImageArray* output, ImageArray* deriv)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			deriv->pixels[i][j] = 1;
		}
	}
}

void _img_activation_relu_deriv(ImageArray* pre_activation, ImageArray* output, ImageArray* deriv)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			output->pixels[i][j] = (pre_activation->pixels[i][j] > 0 ? 1 : 0);
		}
	}
}

