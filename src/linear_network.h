#pragma once
#include "layers/linear/linear_layer.h"

#define MAX_LINEAR_NETWORK_LAYERS 100

typedef struct LinearNetwork
{
	LinearLayer** linlays;
	int num_layers;
	int max_layer_size;
} LinearNetwork;

LinearNetwork* linnet_alloc(void);

void linnet_free(LinearNetwork* linnet);

__boolean linnet_append_linlay(LinearNetwork* linnet, LinearLayer* linlay);

