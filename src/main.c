#include "vanilla_sgd.h"
#include "linear_network.h"
#include "general_ml/loss.h"
#include "general_lib/random.h"
#include <stdio.h>

int main()
{
	int size = 10;
	LinearLayer *linlay = linlay_alloc(size, size, ACT_IDENTITY);
	linlay_randomize_weights(linlay, RNG_XAVIER);
	linlay_set_bias_zero(linlay);
	LinearLayerSGD *linlay_sgd = linlay_sgd_alloc(linlay);
	double *input = _mem_alloc(sizeof(double) * size);
	double *expected = _mem_alloc(sizeof(double) * size);
	double *grad_loss_output = _mem_alloc(sizeof(double) * size);

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
	return 0;
}


