#include "img_structures.h"
#include "img_activation.h"
#include "../../general_lib/random.h"

typedef struct ConvolutionalFilter
{
	double** weights;
	int filter_m;
	int filter_n;
	int input_index;
} ConvolutionalFilter;

ConvolutionalFilter* cnn_filter_alloc(int filter_m, int filter_n, int input_index);

void cnn_filter_free(ConvolutionalFilter* cnn_filter);

void cnn_filter_randomize_weights(ConvolutionalFilter* cnn_filter, double stdev);

double cnn_filter_forward_dot_product(ConvolutionalFilter* cnn_filter, ImageArray* input, int i, int j);

void cnn_filter_forward(ConvolutionalFilter* cnn_filter, ImageLayer* input, ImageArray* output);

typedef struct ConvolutionFilterIndexGrad
{
	double** grad_prenet_weights;
	double** grad_loss_weights;
	double** grad_prenet_input;
	double** grad_loss_input;
	int index_i;
	int index_j;
	int filter_m;
	int filter_n;
} ConvolutionalFilterIndexGrad;

ConvolutionalFilterIndexGrad* cnn_filteri_grad_alloc(ConvolutionalFilter* cnn_filter, int index_i, int index_j);

void cnn_filteri_grad_free(ConvolutionalFilterIndexGrad* cnn_filteri_grad);

void cnn_filteri_grad_compute_weights(
	ConvolutionalFilterIndexGrad* cnn_filteri_grad,
	ConvolutionalFilter* cnn_filter,
	ImageArray* input,
	ImageArray* grad_loss_prenet);

void cnn_filteri_grad_compute_input(
	ConvolutionalFilterIndexGrad* cnn_filteri_grad,
	ConvolutionalFilter* cnn_filter,
	ImageArray* grad_loss_prenet);

void cnn_filteri_grad_compute(
	ConvolutionalFilterIndexGrad* cnn_filteri_grad,
	ConvolutionalFilter* cnn_filter,
	ImageArray* input,
	ImageArray* grad_loss_prenet);

typedef struct ConvolutionalFilterGrad
{
	ConvolutionalFilterIndexGrad*** filteri_grads;
	int input_m;
	int input_n;
	int filter_m;
	int filter_n;
	int output_m;
	int output_n;
} ConvolutionalFilterGrad;

ConvolutionalFilterGrad* cnn_filter_grad_alloc(ConvolutionalFilter* cnn_filter, int output_m, int output_n);

void cnn_filter_grad_free(ConvolutionalFilterGrad* cnn_filter_grad);

void cnn_filter_grad_compute(
	ConvolutionalFilterGrad* cnn_filter_grad,
	ConvolutionalFilter* cnn_filter,
	ImageArray* input,
	ImageArray* grad_loss_prenet);

typedef struct ConvolutionalMultiFilter
{
	ConvolutionalFilter** filters;
	int num_filters;
	int* filter_indexes;
	int filter_m;
	int filter_n;
	int input_m;
	int input_n;
	int output_m;
	int output_n;
	double bias;
	__img_activation_type act_type;
} ConvolutionalMultiFilter;

ConvolutionalMultiFilter* cnn_mfilter_alloc(
	int num_filters,
	int* filter_indexes,
	int filter_m,
	int filter_n,
	int input_m,
	int input_n,
	__img_activation_type act_type);

void cnn_mfilter_free(ConvolutionalMultiFilter* cnn_mfilter);

void cnn_mfilter_randomize_weights(ConvolutionalMultiFilter* cnn_filter, double stdev);

void cnn_mfilter_set_bias_zero(ConvolutionalMultiFilter* cnn_mfilter);

typedef struct ConvolutionalMultiFilterEvaluation
{
	ImageArray** prenet;
	int num_filters;
	ImageArray* pre_activation;
	ImageArray* output;
	int m;
	int n;
} ConvolutionalMultiFilterEvaluation;

ConvolutionalMultiFilterEvaluation* cnn_mfilter_eval_alloc(ConvolutionalMultiFilter* cnn_mfilter, ImageArray* output);

void cnn_mfilter_eval_free(ConvolutionalMultiFilterEvaluation* cnn_mfilter_eval);

void cnn_mfilter_eval_compute(
	ConvolutionalMultiFilterEvaluation* cnn_mfilter_eval,
	ConvolutionalMultiFilter* cnn_mfilter,
	ImageLayer* input);

typedef struct ConvolutionalMultiFilterGrad
{
	ConvolutionalFilterGrad** filter_grads;
	ImageArray** grad_net_prenet;
	ImageArray** grad_loss_prenet;
	ImageArray** grad_loss_input;
	int num_filters;
	ImageArray* grad_out_net;
	ImageArray* grad_loss_net;
	double grad_net_bias;
	ImageArray* grad_loss_bias;
	int input_m;
	int input_n;
	int output_m;
	int output_n;
} ConvolutionalMultiFilterGrad;

ConvolutionalMultiFilterGrad* cnn_mfilter_grad_alloc(ConvolutionalMultiFilter* cnn_mfilter);

void cnn_mfilter_grad_free(ConvolutionalMultiFilterGrad* cnn_mfilter_grad);

void cnn_mfilter_grad_compute_net(
	ConvolutionalMultiFilterGrad* cnn_mfilter_grad,
	ConvolutionalMultiFilter* cnn_mfilter,
	ConvolutionalMultiFilterEvaluation* cnn_mfilter_eval,
	ImageArray* grad_loss_out);

void cnn_mfilter_grad_compute_bias(ConvolutionalMultiFilterGrad* cnn_mfilter_grad);

void cnn_mfilter_grad_compute_prenet(ConvolutionalMultiFilterGrad* cnn_mfilter_grad);

void cnn_mfilter_grad_compute_filters(
	ConvolutionalMultiFilterGrad* cnn_mfilter_grad,
	ConvolutionalMultiFilter* cnn_mfilter,
	ImageArray* input);

void cnn_mfilter_grad_compute_input_at_index(
	ImageArray* grad_loss_input,
	ConvolutionalFilterGrad* cnn_filter_grad,
	int input_i,
	int input_j);

void cnn_mfilter_grad_compute_input_prenet(ImageArray* grad_loss_input, ConvolutionalFilterGrad* cnn_filter_grad, ConvolutionalMultiFilter* cnn_mfilter);

void cnn_mfilter_grad_compute_input(ConvolutionalMultiFilterGrad* cnn_mfilter_grad, ConvolutionalMultiFilter* cnn_mfilter);

void cnn_mfilter_grad_compute(
	ConvolutionalMultiFilterGrad* cnn_mfilter_grad,
	ConvolutionalMultiFilter* cnn_mfilter,
	ConvolutionalMultiFilterEvaluation* cnn_mfilter_eval,
	ImageArray* input,
	ImageArray* grad_loss_out);
