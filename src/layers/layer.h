#include "linear/linear_layer.h"

typedef int __layer_type;

#define LAYER_LINEAR_LAYER 0

typedef struct Layer
{
	void* layer;
	__layer_type layer_type;
} Layer;