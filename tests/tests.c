#include "tests.h"

void test_linear_sgd(void)
{
	int size = 10;
	LinearLayer* linlay = linlay_alloc(size, size, ACT_IDENTITY);
	linlay_randomize_weights(linlay, RNG_XAVIER);
	linlay_set_bias_zero(linlay);
	LinearLayerSGD* linlay_sgd = linlay_sgd_alloc(linlay);
	double* input = _mem_alloc(sizeof(double) * size);
	double* expected = _mem_alloc(sizeof(double) * size);
	double* grad_loss_output = _mem_alloc(sizeof(double) * size);

	for (int i = 0; i < 1000; ++i)
	{
		int index = round(_rand_between(0, size - 1));
		input[index] = _rand_between(0, 3);
		expected[index] = 1;
		for (int j = 0; j < size; ++j)
		{
			if (j != index)
			{
				input[j] = _rand_between(-3, 0);
				expected[j] = 0;
			}
		}
		linlay_sgd_forward(linlay_sgd, input);
		_loss_deriv(linlay_sgd->eval->output, expected, grad_loss_output, 1, LOSS_MSE);
		linlay_sgd_backward(linlay_sgd, input, grad_loss_output, 0.01);
		printf("%f\n", linlay_sgd->eval->output[index]);
	}

	free(input);
	free(expected);
	free(grad_loss_output);
	linlay_sgd_free(linlay_sgd);
	linlay_free(linlay);
}

void test_cnn_forward(void)
{
	int NUM_FILTERS = 2;
	int INPUT_NUM_IMG = 3;
	int INPUT_M = 4;
	int INPUT_N = 4;
	int FILTER_M = 2;
	int FILTER_N = 2;
	int OUTPUT_M = INPUT_M - FILTER_M + 1;
	int OUTPUT_N = INPUT_N - FILTER_N + 1;
	int* NUM_INDEXES = _mem_alloc(sizeof(int) * NUM_FILTERS);
	NUM_INDEXES[0] = 1;
	NUM_INDEXES[1] = 3;
	int** INDEXES = _mem_alloc(sizeof(int*) * NUM_FILTERS);
	INDEXES[0] = _mem_alloc(sizeof(int) * NUM_INDEXES[0]);
	INDEXES[1] = _mem_alloc(sizeof(int) * NUM_INDEXES[1]);
	INDEXES[0][0] = 1;
	INDEXES[1][0] = 0;
	INDEXES[1][1] = 1;
	INDEXES[1][2] = 2;
	ConvolutionalLayer* cnn_layer = cnn_layer_alloc(NUM_FILTERS, INPUT_M, INPUT_N, INDEXES, NUM_INDEXES);
	cnn_layer_randomize_weights(cnn_layer, RNG_HE);
	cnn_layer_set_bias_zero(cnn_layer);
	ImageLayer* input = img_layer_alloc(INPUT_NUM_IMG, INPUT_M, INPUT_N);
	ImageLayer* output = img_layer_alloc(NUM_FILTERS, OUTPUT_M, OUTPUT_N);
	input->images[0]->pixels[0] = 1;
	input->images[0]->pixels[1] = 1;
	input->images[0]->pixels[2] = 1;
	input->images[0]->pixels[3] = 1;
	input->images[0]->pixels[4] = 1;
	input->images[0]->pixels[5] = 1;
	input->images[0]->pixels[6] = 1;
	input->images[0]->pixels[7] = 1;
	input->images[0]->pixels[8] = 1;
	input->images[0]->pixels[9] = 1;
	input->images[0]->pixels[10] = 1;
	input->images[0]->pixels[11] = 1;
	input->images[0]->pixels[12] = 1;
	input->images[0]->pixels[13] = 1;
	input->images[0]->pixels[14] = 1;
	input->images[0]->pixels[15] = 1;
	input->images[1]->pixels[0] = 1;
	input->images[1]->pixels[1] = 1;
	input->images[1]->pixels[2] = 1;
	input->images[1]->pixels[3] = 1;
	input->images[1]->pixels[4] = 1;
	input->images[1]->pixels[5] = 1;
	input->images[1]->pixels[6] = 1;
	input->images[1]->pixels[7] = 1;
	input->images[1]->pixels[8] = 1;
	input->images[1]->pixels[9] = 1;
	input->images[1]->pixels[10] = 1;
	input->images[1]->pixels[11] = 1;
	input->images[1]->pixels[12] = 1;
	input->images[1]->pixels[13] = 1;
	input->images[1]->pixels[14] = 1;
	input->images[1]->pixels[15] = 1;
	input->images[2]->pixels[0] = 1;
	input->images[2]->pixels[1] = 1;
	input->images[2]->pixels[2] = 1;
	input->images[2]->pixels[3] = 1;
	input->images[2]->pixels[4] = 1;
	input->images[1]->pixels[5] = 1;
	input->images[2]->pixels[6] = 1;
	input->images[2]->pixels[7] = 1;
	input->images[2]->pixels[8] = 1;
	input->images[2]->pixels[9] = 1;
	input->images[2]->pixels[10] = 1;
	input->images[2]->pixels[11] = 1;
	input->images[2]->pixels[12] = 1;
	input->images[2]->pixels[13] = 1;
	input->images[2]->pixels[14] = 1;
	input->images[2]->pixels[15] = 1;

	for (int i = 0; i < 1000; ++i)
	{
		cnn_layer_forward(cnn_layer, input, output);
	}

	free(NUM_INDEXES);
	free(INDEXES[0]);
	free(INDEXES[1]);
	free(INDEXES);
	cnn_layer_free(cnn_layer);
	img_layer_free(input);
	img_layer_free(output);
}
