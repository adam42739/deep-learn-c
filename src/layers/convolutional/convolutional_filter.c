#include "convolutional_filter.h"

ConvolutionalFilter *cnn_filter_alloc(int filter_m, int filter_n, int input_index)
{
	ConvolutionalFilter *cnn_filter = _mem_alloc(sizeof(ConvolutionalFilter));
	cnn_filter->weights = _mem_alloc(sizeof(double *) * filter_m);
	for (int i = 0; i < filter_m; ++i)
	{
		cnn_filter->weights[i] = _mem_alloc(sizeof(double) * filter_n);
	}
	cnn_filter->filter_m = filter_m;
	cnn_filter->filter_n = filter_n;
	cnn_filter->input_index = input_index;
	return cnn_filter;
}

void cnn_filter_free(ConvolutionalFilter *cnn_filter)
{
	for (int i = 0; i < cnn_filter->filter_m; ++i)
	{
		free(cnn_filter->weights[i]);
	}
	free(cnn_filter->weights);
	free(cnn_filter);
}

void cnn_filter_randomize_weights(ConvolutionalFilter *cnn_filter, double stdev)
{
	for (int i = 0; i < cnn_filter->filter_m; ++i)
	{
		for (int j = 0; j < cnn_filter->filter_n; ++j)
		{
			cnn_filter->weights[i][j] = _rng_normal(0, stdev);
		}
	}
}

double cnn_filter_forward_dot_product(ConvolutionalFilter *cnn_filter, ImageArray *input, int i, int j)
{
	double sumprod = 0;
	for (int k = 0; k < cnn_filter->filter_m; ++k)
	{
		for (int l = 0; l < cnn_filter->filter_n; ++l)
		{
			sumprod += cnn_filter->weights[k][l] * input->pixels[i + k][j + l];
		}
	}
	return sumprod;
}

void cnn_filter_forward(ConvolutionalFilter *cnn_filter, ImageLayer *input, ImageArray *output)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			output->pixels[i][j] = cnn_filter_forward_dot_product(cnn_filter, input->img_arrays[cnn_filter->input_index], i, j);
		}
	}
}

ConvolutionalFilterIndexGrad *cnn_filteri_grad_alloc(ConvolutionalFilter *cnn_filter, int index_i, int index_j)
{
	ConvolutionalFilterIndexGrad *cnn_filteri_grad = _mem_alloc(sizeof(ConvolutionalFilterIndexGrad));
	cnn_filteri_grad->grad_prenet_weights = _mem_alloc(sizeof(double *) * cnn_filter->filter_m);
	cnn_filteri_grad->grad_loss_weights = _mem_alloc(sizeof(double *) * cnn_filter->filter_m);
	cnn_filteri_grad->grad_prenet_input = _mem_alloc(sizeof(double *) * cnn_filter->filter_m);
	cnn_filteri_grad->grad_loss_input = _mem_alloc(sizeof(double *) * cnn_filter->filter_m);
	for (int i = 0; i < cnn_filter->filter_m; ++i)
	{
		cnn_filteri_grad->grad_prenet_weights[i] = _mem_alloc(sizeof(double) * cnn_filter->filter_n);
		cnn_filteri_grad->grad_loss_weights[i] = _mem_alloc(sizeof(double) * cnn_filter->filter_n);
		cnn_filteri_grad->grad_prenet_input[i] = _mem_alloc(sizeof(double) * cnn_filter->filter_n);
		cnn_filteri_grad->grad_loss_input[i] = _mem_alloc(sizeof(double) * cnn_filter->filter_n);
	}
	cnn_filteri_grad->index_i = index_i;
	cnn_filteri_grad->index_j = index_j;
	cnn_filteri_grad->filter_m = cnn_filter->filter_m;
	cnn_filteri_grad->filter_n = cnn_filter->filter_n;
	return cnn_filteri_grad;
}

void cnn_filteri_grad_free(ConvolutionalFilterIndexGrad *cnn_filteri_grad)
{
	for (int i = 0; i < cnn_filteri_grad->filter_m; ++i)
	{
		free(cnn_filteri_grad->grad_prenet_weights[i]);
		free(cnn_filteri_grad->grad_loss_weights[i]);
		free(cnn_filteri_grad->grad_prenet_input[i]);
		free(cnn_filteri_grad->grad_loss_input[i]);
	}
	free(cnn_filteri_grad->grad_prenet_weights);
	free(cnn_filteri_grad->grad_loss_weights);
	free(cnn_filteri_grad->grad_prenet_input);
	free(cnn_filteri_grad->grad_loss_input);
	free(cnn_filteri_grad);
}

void cnn_filteri_grad_compute_weights(
	ConvolutionalFilterIndexGrad *cnn_filteri_grad,
	ConvolutionalFilter *cnn_filter,
	ImageArray *input,
	ImageArray *grad_loss_prenet)
{
	for (int i = 0; i < cnn_filter->filter_m; ++i)
	{
		for (int j = 0; j < cnn_filter->filter_n; ++j)
		{
			int i_o = cnn_filteri_grad->index_i;
			int j_o = cnn_filteri_grad->index_j;
			int i_i = cnn_filteri_grad->index_i + i;
			int j_i = cnn_filteri_grad->index_j + j;
			cnn_filteri_grad->grad_prenet_weights[i][j] = input->pixels[i_i][j_i];
			cnn_filteri_grad->grad_loss_weights[i][j] = cnn_filteri_grad->grad_prenet_weights[i][j] * grad_loss_prenet->pixels[i_o][j_o];
		}
	}
}

void cnn_filteri_grad_compute_input(
	ConvolutionalFilterIndexGrad *cnn_filteri_grad,
	ConvolutionalFilter *cnn_filter,
	ImageArray *grad_loss_prenet)
{
	for (int i = 0; i < cnn_filter->filter_m; ++i)
	{
		for (int j = 0; j < cnn_filter->filter_n; ++j)
		{
			int i_o = cnn_filteri_grad->index_i;
			int j_o = cnn_filteri_grad->index_j;
			cnn_filteri_grad->grad_prenet_input[i][j] = cnn_filter->weights[i][j];
			cnn_filteri_grad->grad_loss_input[i][j] = cnn_filteri_grad->grad_prenet_input[i][j] * grad_loss_prenet->pixels[i_o][j_o];
		}
	}
}

void cnn_filteri_grad_compute(
	ConvolutionalFilterIndexGrad *cnn_filteri_grad,
	ConvolutionalFilter *cnn_filter,
	ImageArray *input,
	ImageArray *grad_loss_prenet)
{
	cnn_filteri_grad_compute_weights(cnn_filteri_grad, cnn_filter, input, grad_loss_prenet);
	cnn_filteri_grad_compute_input(cnn_filteri_grad, cnn_filter, grad_loss_prenet);
}

ConvolutionalFilterGrad *cnn_filter_grad_alloc(ConvolutionalFilter *cnn_filter, int output_m, int output_n)
{
	ConvolutionalFilterGrad *cnn_filter_grad = _mem_alloc(sizeof(ConvolutionalFilterGrad));
	cnn_filter_grad->filteri_grads = _mem_alloc(sizeof(ConvolutionalFilterIndexGrad **) * output_m);
	for (int i = 0; i < output_m; ++i)
	{
		cnn_filter_grad->filteri_grads[i] = _mem_alloc(sizeof(ConvolutionalFilterIndexGrad *) * output_n);
		for (int j = 0; j < output_n; ++j)
		{
			cnn_filter_grad->filteri_grads[i][j] = cnn_filteri_grad_alloc(cnn_filter, i, j);
		}
	}
	cnn_filter_grad->output_m = output_m;
	cnn_filter_grad->output_n = output_n;
	return cnn_filter_grad;
}

void cnn_filter_grad_free(ConvolutionalFilterGrad *cnn_filter_grad)
{
	for (int i = 0; i < cnn_filter_grad->output_m; ++i)
	{
		for (int j = 0; j < cnn_filter_grad->output_n; ++j)
		{
			cnn_filteri_grad_free(cnn_filter_grad->filteri_grads[i][j]);
		}
		free(cnn_filter_grad->filteri_grads[i]);
	}
	free(cnn_filter_grad->filteri_grads);
	free(cnn_filter_grad);
}

void cnn_filter_grad_compute(
	ConvolutionalFilterGrad *cnn_filter_grad,
	ConvolutionalFilter *cnn_filter,
	ImageArray *input,
	ImageArray *grad_loss_prenet)
{
	for (int i = 0; i < cnn_filter_grad->output_m; ++i)
	{
		for (int j = 0; j < cnn_filter_grad->output_n; ++j)
		{
			cnn_filteri_grad_compute(cnn_filter_grad->filteri_grads[i][j], cnn_filter, input, grad_loss_prenet);
		}
	}
}

ConvolutionalMultiFilter *cnn_mfilter_alloc(
	int num_filters,
	int *filter_indexes,
	int filter_m,
	int filter_n,
	int input_m,
	int input_n,
	__img_activation_type act_type)
{
	ConvolutionalMultiFilter *cnn_mfilter = _mem_alloc(sizeof(ConvolutionalMultiFilter));
	cnn_mfilter->filters = _mem_alloc(sizeof(ConvolutionalFilter *) * num_filters);
	for (int i = 0; i < num_filters; ++i)
	{
		cnn_mfilter->filters[i] = cnn_filter_alloc(filter_m, filter_n, filter_indexes[i]);
	}
	cnn_mfilter->num_filters = num_filters;
	cnn_mfilter->filter_indexes = _mem_alloc(sizeof(int) * num_filters);
	memcpy(cnn_mfilter->filter_indexes, filter_indexes, sizeof(int) * num_filters);
	cnn_mfilter->filter_m = filter_m;
	cnn_mfilter->filter_n = filter_n;
	cnn_mfilter->input_m = input_m;
	cnn_mfilter->input_n = input_n;
	cnn_mfilter->output_m = input_m - filter_m + 1;
	cnn_mfilter->output_n = input_n - filter_n + 1;
	cnn_mfilter->act_type = act_type;
	return cnn_mfilter;
}

void cnn_mfilter_free(ConvolutionalMultiFilter *cnn_mfilter)
{
	for (int i = 0; i < cnn_mfilter->num_filters; ++i)
	{
		cnn_filter_free(cnn_mfilter->filters[i]);
	}
	free(cnn_mfilter->filters);
	free(cnn_mfilter);
}

void cnn_mfilter_randomize_weights(ConvolutionalMultiFilter *cnn_filter, double stdev)
{
	for (int i = 0; i < cnn_filter->num_filters; ++i)
	{
		cnn_filter_randomize_weights(cnn_filter->filters[i], stdev);
	}
}

void cnn_mfilter_set_bias_zero(ConvolutionalMultiFilter *cnn_mfilter)
{
	cnn_mfilter->bias = 0;
}

ConvolutionMultiFilterEvaluation *cnn_mfilter_eval_alloc(ConvolutionalMultiFilter *cnn_mfilter)
{
	ConvolutionMultiFilterEvaluation *cnn_mfilter_eval = _mem_alloc(sizeof(ConvolutionMultiFilterEvaluation));
	cnn_mfilter_eval->prenet = _mem_alloc(sizeof(ImageArray *) * cnn_mfilter->num_filters);
	for (int i = 0; i < cnn_mfilter->num_filters; ++i)
	{
		cnn_mfilter_eval->prenet[i] = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
	}
	cnn_mfilter_eval->num_filters = cnn_mfilter->num_filters;
	cnn_mfilter_eval->pre_activation = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
	cnn_mfilter_eval->output = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
	cnn_mfilter_eval->m = cnn_mfilter->output_m;
	cnn_mfilter_eval->n = cnn_mfilter->output_n;
	return cnn_mfilter_eval;
}

void cnn_mfilter_eval_free(ConvolutionMultiFilterEvaluation *cnn_mfilter_eval)
{
	for (int i = 0; i < cnn_mfilter_eval->num_filters; ++i)
	{
		img_array_free(cnn_mfilter_eval->prenet[i]);
	}
	free(cnn_mfilter_eval->prenet);
	img_array_free(cnn_mfilter_eval->pre_activation);
	img_array_free(cnn_mfilter_eval->output);
	free(cnn_mfilter_eval);
}

void cnn_mfilter_eval_compute(
	ConvolutionMultiFilterEvaluation *cnn_mfilter_eval,
	ConvolutionalMultiFilter *cnn_mfilter,
	ImageLayer *input)
{
	for (int i = 0; i < cnn_mfilter->num_filters; ++i)
	{
		cnn_filter_forward(cnn_mfilter->filters[i], input, cnn_mfilter_eval->prenet[i]);
	}
	img_array_sum(cnn_mfilter_eval->prenet, cnn_mfilter_eval->num_filters, cnn_mfilter_eval->pre_activation);
	_img_activation(cnn_mfilter_eval->pre_activation, cnn_mfilter_eval->output, cnn_mfilter->act_type);
}

ConvolutionalMultiFilterGrad *cnn_mfilter_grad_alloc(ConvolutionalMultiFilter *cnn_mfilter)
{
	ConvolutionalMultiFilterGrad *cnn_mfilter_grad = _mem_alloc(sizeof(ConvolutionalFilterGrad));
	cnn_mfilter_grad->grad_net_prenet = _mem_alloc(sizeof(ImageArray *) * cnn_mfilter->num_filters);
	cnn_mfilter_grad->grad_loss_prenet = _mem_alloc(sizeof(ImageArray *) * cnn_mfilter->num_filters);
	cnn_mfilter_grad->grad_prenet_input = _mem_alloc(sizeof(ImageArray *) * cnn_mfilter->num_filters);
	cnn_mfilter_grad->grad_loss_input = _mem_alloc(sizeof(ImageArray *) * cnn_mfilter->num_filters);
	for (int i = 0; i < cnn_mfilter->num_filters; ++i)
	{
		cnn_mfilter_grad->grad_net_prenet[i] = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
		cnn_mfilter_grad->grad_loss_prenet[i] = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
		cnn_mfilter_grad->grad_prenet_input[i] = img_array_alloc(cnn_mfilter->input_m, cnn_mfilter->input_n);
		cnn_mfilter_grad->grad_loss_input[i] = img_array_alloc(cnn_mfilter->input_m, cnn_mfilter->input_n);
	}
	cnn_mfilter_grad->num_filters = cnn_mfilter->num_filters;
	cnn_mfilter_grad->grad_out_net = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
	cnn_mfilter_grad->grad_loss_net = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
	cnn_mfilter_grad->grad_loss_bias = img_array_alloc(cnn_mfilter->output_m, cnn_mfilter->output_n);
	return cnn_mfilter_grad;
}

void cnn_mfilter_grad_free(ConvolutionalMultiFilterGrad *cnn_mfilter_grad)
{
	for (int i = 0; i < cnn_mfilter_grad->num_filters; ++i)
	{
		img_array_free(cnn_mfilter_grad->grad_net_prenet[i]);
		img_array_free(cnn_mfilter_grad->grad_loss_prenet[i]);
		img_array_free(cnn_mfilter_grad->grad_prenet_input[i]);
		img_array_free(cnn_mfilter_grad->grad_loss_input[i]);
	}
	free(cnn_mfilter_grad->grad_net_prenet);
	free(cnn_mfilter_grad->grad_loss_prenet);
	free(cnn_mfilter_grad->grad_prenet_input);
	free(cnn_mfilter_grad->grad_loss_input);
	img_array_free(cnn_mfilter_grad->grad_out_net);
	img_array_free(cnn_mfilter_grad->grad_loss_net);
	img_array_free(cnn_mfilter_grad->grad_loss_bias);
	free(cnn_mfilter_grad);
}

void cnn_mfilter_grad_compute_net(
	ConvolutionalMultiFilterGrad *cnn_mfilter_grad,
	ConvolutionalMultiFilter *cnn_mfilter,
	ConvolutionMultiFilterEvaluation *cnn_mfilter_eval,
	ImageArray *grad_loss_out)
{
	_img_activation_deriv(cnn_mfilter_eval->pre_activation, cnn_mfilter_eval->output, cnn_mfilter_grad->grad_out_net, cnn_mfilter->act_type);
	img_array_product(cnn_mfilter_grad->grad_out_net, grad_loss_out, cnn_mfilter_grad->grad_loss_net);
}

void cnn_mfilter_grad_compute_bias(ConvolutionalMultiFilterGrad *cnn_mfilter_grad)
{
	cnn_mfilter_grad->grad_net_bias = 1;
	img_array_copy(cnn_mfilter_grad->grad_loss_bias, cnn_mfilter_grad->grad_loss_net);
}

void cnn_mfilter_grad_compute_prenet(ConvolutionalMultiFilterGrad *cnn_mfilter_grad)
{
	img_array_set(cnn_mfilter_grad->grad_net_prenet, 1);
	img_array_copy(cnn_mfilter_grad->grad_loss_prenet, cnn_mfilter_grad->grad_loss_net);
}

void cnn_mfilter_grad_compute_filters(
	ConvolutionalMultiFilterGrad *cnn_mfilter_grad,
	ConvolutionalMultiFilter *cnn_mfilter,
	ImageArray *input)
{
	for (int i = 0; i < cnn_mfilter_grad->num_filters; ++i)
	{
		cnn_filter_grad_compute(cnn_mfilter_grad->filter_grads[i], cnn_mfilter->filters[i], input, cnn_mfilter_grad->grad_loss_prenet);
	}
}

void cnn_mfilter_grad_compute_input(ConvolutionalMultiFilterGrad *cnn_mfilter_grad)
{
	for (int i = 0; i < cnn_mfilter_grad->filter_grads; ++i)
	{
		// TODO
		// only specific inputs affect specific outputs
	}
}

void cnn_mfilter_grad_compute(
	ConvolutionalMultiFilterGrad *cnn_mfilter_grad,
	ConvolutionalMultiFilter *cnn_mfilter,
	ConvolutionMultiFilterEvaluation *cnn_mfilter_eval,
	ImageArray *input,
	ImageArray *grad_loss_out)
{
	cnn_mfilter_grad_compute_net(cnn_mfilter_grad, cnn_mfilter, cnn_mfilter_eval, grad_loss_out);
	cnn_mfilter_grad_compute_bias(cnn_mfilter_grad);
	cnn_mfilter_grad_compute_prenet(cnn_mfilter_grad);
	cnn_mfilter_grad_compute_filters(cnn_mfilter_grad, cnn_mfilter, input);
	cnn_mfilter_grad_compute_input(cnn_mfilter_grad);
}
