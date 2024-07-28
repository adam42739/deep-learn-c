#include "linear_network.h"

LinearNetwork *linnet_alloc(void)
{
	LinearNetwork *linnet = _mem_alloc(sizeof(LinearNetwork));
	linnet->linlays = _mem_alloc(sizeof(LinearLayer *) * MAX_LINEAR_NETWORK_LAYERS);
	linnet->num_layers = 0;
	linnet->max_layer_size = 0;
	return linnet;
}

void linnet_free(LinearNetwork *linnet)
{
	for (int i = 0; i < linnet->num_layers; ++i)
	{
		linlay_free(linnet->linlays[i]);
	}
	free(linnet->linlays);
	free(linnet);
}

__boolean linnet_append_linlay(LinearNetwork *linnet, LinearLayer *linlay)
{
	if (linnet->num_layers < MAX_LINEAR_NETWORK_LAYERS)
	{
		linnet->linlays[linnet->num_layers] = linlay;
		++linnet->num_layers;
		if (linlay->num_linmods > linnet->max_layer_size)
		{
			linnet->max_layer_size = linlay->num_linmods;
		}
		return TRUE;
	}
	else
	{
		return FALSE;
	}
}
