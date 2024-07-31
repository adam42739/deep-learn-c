#include "../convolutional/img_structures.h"
#include <float.h>

typedef int __pool_layer_type;

#define POOL_LAYER_MAX 0
#define POOL_LAYER_AVG 1

double _pooling_pool(ImageArray* input, int index_m, int index_n, int filter_m, int filter_n, __pool_layer_type pool_type);

double _pooling_pool_max(ImageArray* input, int index_m, int index_n, int filter_m, int filter_n);

double _pooling_pool_avg(ImageArray* input, int index_m, int index_n, int filter_m, int filter_n);

typedef struct PoolingLayer
{
	int num_input;
	int filter_m;
	int filter_n;
	int stride_m;
	int stride_n;
	int input_m;
	int input_n;
	int output_m;
	int output_n;
	__pool_layer_type pool_type;
} PoolingLayer;

PoolingLayer* pool_layer_alloc(
	int num_input,
	int filter_m,
	int filter_n,
	int stride_m,
	int stride_n,
	int input_m,
	int input_n,
	__pool_layer_type pool_type);

void pool_layer_free(PoolingLayer* pool_layer);

void pool_layer_forward_image(PoolingLayer* pool_layer, ImageArray* input, ImageArray* output);

void pool_layer_forward(PoolingLayer* pool_layer, ImageLayer* input, ImageLayer* output);

typedef struct PoolingLayerEval
{
	ImageLayer* output;
} PoolingLayerEval;

PoolingLayerEval* pool_layer_eval_alloc(PoolingLayer* pool_layer);

void pool_layer_eval_free(PoolingLayerEval* pool_layer_eval);

typedef struct PoolingLayerGrad
{
	int num_input;
	int filter_m;
	int filter_n;
	int stride_m;
	int stride_n;
	int input_m;
	int input_n;
	int output_m;
	int output_n;
	__pool_layer_type pool_type;
	ImageLayer* grad_loss_input;
} PoolingLayerGrad;


PoolingLayerGrad* pool_layer_grad_alloc(PoolingLayer* pool_layer);

void pool_layer_grad_free(PoolingLayerGrad* pool_layer_grad);

void pool_layer_grad_compute(
	PoolingLayerGrad* pool_layer_grad,
	PoolingLayerEval* pool_layer_eval,
	PoolingLayer* pool_layer,
	ImageLayer* input,
	ImageLayer* grad_loss_out);
