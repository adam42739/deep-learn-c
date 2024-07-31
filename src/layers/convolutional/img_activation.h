#include "img_structures.h"

typedef int __img_activation_type;

#define IMG_ACT_IDENTITY 0
#define IMG_ACT_RELU 1

void _img_activation(ImageArray* pre_activation, ImageArray* output, __img_activation_type act_type);

void _img_activation_identity(ImageArray* pre_activation, ImageArray* output);

void _img_activation_relu(ImageArray* pre_activation, ImageArray* output);

void _img_activation_deriv(ImageArray* pre_activation, ImageArray* output, ImageArray* deriv, __img_activation_type act_type);

void _img_activation_identity_deriv(ImageArray* pre_activation, ImageArray* output, ImageArray* deriv);

void _img_activation_relu_deriv(ImageArray* pre_activation, ImageArray* output, ImageArray* deriv);
