#include "pooling_layer.h"

double _pooling_pool(ImageArray* input, int index_m, int index_n, int filter_m, int filter_n, __pool_layer_type pool_type)
{
	switch (pool_type)
	{
	case POOL_LAYER_MAX:
		return _pooling_pool_max(input, index_m, index_n, filter_m, filter_n);
	case POOL_LAYER_AVG:
		return _pooling_pool_avg(input, index_m, index_n, filter_m, filter_n);
	default:
		return 0;
	}
}

double _pooling_pool_max(ImageArray* input, int index_m, int index_n, int filter_m, int filter_n)
{
	double max = -DBL_MAX;
	for (int i = index_m; i < index_m + filter_m; ++i)
	{
		for (int j = index_n; j < index_n + filter_n; ++j)
		{
			if (input->pixels[i][j] > max)
			{
				max = input->pixels[i][j];
			}
		}
	}
	return max;
}

double _pooling_pool_avg(ImageArray* input, int index_m, int index_n, int filter_m, int filter_n)
{
	double value = 0;
	for (int i = index_m; i < index_m + filter_m; ++i)
	{
		for (int j = index_n; j < index_n + filter_n; ++j)
		{
			value += input->pixels[i][j];
		}
	}
	double factor = filter_m * filter_n;
	return value / factor;
}


PoolingLayer* pool_layer_alloc(
	int num_input,
	int filter_m,
	int filter_n,
	int stride_m,
	int stride_n,
	int input_m,
	int input_n,
	__pool_layer_type pool_type)
{
	PoolingLayer* pool_layer = _mem_alloc(sizeof(PoolingLayer));
	pool_layer->num_input = num_input;
	pool_layer->filter_m = filter_m;
	pool_layer->filter_n = filter_n;
	pool_layer->input_m = input_m;
	pool_layer->input_n = input_n;
	pool_layer->stride_m = stride_m;
	pool_layer->stride_n = stride_n;
	pool_layer->output_m = (int)((input_m - filter_m) / stride_m) + 1;
	pool_layer->output_n = (int)((input_n - filter_n) / stride_n) + 1;
	pool_layer->pool_type = pool_type;
	return pool_layer;
}

void pool_layer_free(PoolingLayer* pool_layer)
{
	free(pool_layer);
}

void pool_layer_forward_image(PoolingLayer* pool_layer, ImageArray* input, ImageArray* output)
{
	int strided_m = 0;
	for (int i = 0; i < pool_layer->output_m; ++i)
	{
		int strided_n = 0;
		for (int j = 0; j < pool_layer->output_n; ++j)
		{
			output->pixels[i][j] = _pooling_pool(input, i, j, strided_m, strided_n, pool_layer->pool_type);
			strided_n += pool_layer->stride_n;
		}
		strided_m += pool_layer->stride_m;
	}
}

void pool_layer_forward(PoolingLayer* pool_layer, ImageLayer* input, ImageLayer* output)
{
	for (int i = 0; i < pool_layer->num_input; ++i)
	{
		pool_layer_forward_image(pool_layer, input->img_arrays[i], output->img_arrays[i]);
	}
}


PoolingLayerEval* pool_layer_eval_alloc(PoolingLayer* pool_layer)
{
	PoolingLayerEval* pool_layer_eval = _mem_alloc(sizeof(PoolingLayerEval));
	pool_layer_eval->output = img_layer_alloc(pool_layer->num_input, pool_layer->output_m, pool_layer->output_n);
	return pool_layer_eval;
}

void pool_layer_eval_free(PoolingLayerEval* pool_layer_eval)
{
	img_layer_free(pool_layer_eval);
	free(pool_layer_eval);
}

PoolingLayerGrad* pool_layer_grad_alloc(PoolingLayer* pool_layer)
{
	PoolingLayerGrad* pool_layer_grad = _mem_alloc(sizeof(PoolingLayerGrad));
	pool_layer_grad->num_input = pool_layer->num_input;
	pool_layer_grad->filter_m = pool_layer->filter_m;
	pool_layer_grad->filter_n = pool_layer->filter_n;
	pool_layer_grad->stride_m = pool_layer->stride_m;
	pool_layer_grad->stride_n = pool_layer->stride_n;
	pool_layer_grad->input_m = pool_layer->input_m;
	pool_layer_grad->input_n = pool_layer->input_n;
	pool_layer_grad->output_m = pool_layer->output_m;
	pool_layer_grad->output_n = pool_layer->output_n;
	pool_layer_grad->pool_type = pool_layer->pool_type;
	pool_layer_grad->grad_loss_input = img_layer_alloc(pool_layer->num_input, pool_layer->input_m, pool_layer->input_n);
	return pool_layer_grad;
}

void pool_layer_grad_free(PoolingLayerGrad* pool_layer_grad)
{
	img_layer_free(pool_layer_grad->grad_loss_input);
	free(pool_layer_grad);
}

double pool_layer_grad_compute_filter_at_index_max(PoolingLayerGrad* pool_layer_grad, ImageArray* input, ImageArray* grad_loss_out, int input_m, int input_n)
{
	return 0;
}

double pool_layer_grad_compute_filter_at_index_avg(PoolingLayerGrad* pool_layer_grad, ImageArray* grad_loss_out, int input_m, int input_n)
{
	double grad_out_input = 0;
	int start_m = (int)((input_m - pool_layer_grad->filter_m) / pool_layer_grad->stride_m);
	if (((input_m - pool_layer_grad->filter_m) % pool_layer_grad->stride_m) != 0)
	{
		++start_m;
	}
	start_m = MIN(MAX(start_m, 0), grad_loss_out->m - 1);
	int end_m = (int)(input_m / pool_layer_grad->stride_m);
	end_m = MIN(MAX(end_m, 0), grad_loss_out->m - 1);
	int start_n = (int)((input_n - pool_layer_grad->filter_n) / pool_layer_grad->stride_n);
	if (((input_n - pool_layer_grad->filter_n) % pool_layer_grad->stride_n) != 0)
	{
		++start_n;
	}
	start_n = MIN(MAX(start_n, 0), grad_loss_out->n - 1);
	int end_n = (int)(input_n / pool_layer_grad->stride_n);
	end_n = MIN(MAX(end_n, 0), grad_loss_out->n - 1);
	for (int i = start_m; i <= end_m; ++i)
	{
		for (int j = start_n; j <= end_n; ++j)
		{
			grad_out_input += grad_loss_out->pixels[i][j];
		}
	}
	double factor = pool_layer_grad->filter_m * pool_layer_grad->filter_n;
	return grad_out_input / factor;
}

double pool_layer_grad_compute_filter_at_index(PoolingLayerGrad* pool_layer_grad, ImageArray* input, ImageArray* grad_loss_out, int input_m, int input_n, __pool_layer_type pool_type)
{
	switch (pool_type)
	{
	case POOL_LAYER_MAX:
		return pool_layer_grad_compute_filter_at_index_max(pool_layer_grad, input, grad_loss_out, input_m, input_n);
	case POOL_LAYER_AVG:
		return pool_layer_grad_compute_filter_at_index_avg(pool_layer_grad, grad_loss_out, input_m, input_n);
	default:
		return 0;
	}
}

void pool_layer_grad_compute_filter(PoolingLayerGrad* pool_layer_grad, ImageArray* input, ImageArray* grad_loss_out, int img_num)
{
	ImageArray* grad_loss_input = pool_layer_grad->grad_loss_input->img_arrays[img_num];
	for (int i = 0; i < pool_layer_grad->input_m; ++i)
	{
		for (int j = 0; j < pool_layer_grad->input_n; ++j)
		{
			grad_loss_input->pixels[i][j] = pool_layer_grad_compute_filter_at_index(pool_layer_grad, input, grad_loss_out, i, j, pool_layer_grad->pool_type);
		}
	}
}

void pool_layer_grad_compute(
	PoolingLayerGrad* pool_layer_grad,
	PoolingLayerEval* pool_layer_eval,
	PoolingLayer* pool_layer,
	ImageLayer* input,
	ImageLayer* grad_loss_out)
{
	for (int i = 0; i < pool_layer_grad->num_input; ++i)
	{
		pool_layer_grad_compute_filter(pool_layer_grad, input->img_arrays[i], grad_loss_out->img_arrays[i], i);
	}
}



