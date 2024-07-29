#include "img_structures.h"

ImageArray *img_array_alloc(int m, int n)
{
	ImageArray *img_array = _mem_alloc(sizeof(ImageArray));
	img_array->pixels = _mem_alloc(sizeof(double *) * m);
	for (int i = 0; i < m; ++i)
	{
		img_array->pixels[i] = _mem_alloc(sizeof(double) * n);
	}
	img_array->m = m;
	img_array->n = n;
	return img_array;
}

void img_array_free(ImageArray *img_array)
{
	for (int i = 0; i < img_array->m; ++i)
	{
		free(img_array->pixels[i]);
	}
	free(img_array->pixels);
	free(img_array);
}

void img_array_sum(ImageArray **img_array, int num_array, ImageArray *output)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			output->pixels[i][j] = 0;
			for (int k = 0; k < num_array; ++k)
			{
				output->pixels[i][j] += img_array[k]->pixels[i][j];
			}
		}
	}
}

void img_array_product(ImageArray *img_array_A, ImageArray *img_array_B, ImageArray *output)
{
	for (int i = 0; i < output->m; ++i)
	{
		for (int j = 0; j < output->n; ++j)
		{
			output->pixels[i][j] = img_array_A->pixels[i][j] * img_array_B->pixels[i][j];
		}
	}
}

void img_array_copy(ImageArray *source, ImageArray *dest)
{
	for (int i = 0; i < dest->m; ++i)
	{
		memcpy(dest->pixels[i], source->pixels[i], sizeof(double) * dest->n);
	}
}

ImageLayer *img_layer_alloc(int num_arrays, int m, int n)
{
	ImageLayer *img_layer = _mem_alloc(sizeof(ImageLayer));
	img_layer->img_arrays = _mem_alloc(sizeof(ImageArray *) * num_arrays);
	for (int i = 0; i < num_arrays; ++i)
	{
		img_layer->img_arrays = img_array_alloc(m, n);
	}
	img_layer->num_arrays = num_arrays;
	img_layer->m = m;
	img_layer->n = n;
	return img_layer;
}

void img_layer_free(ImageLayer *img_layer)
{
	for (int i = 0; i < img_layer->num_arrays; ++i)
	{
		img_array_free(img_layer->img_arrays[i]);
	}
	free(img_layer->img_arrays);
	free(img_layer);
}

void img_array_set(ImageArray *img_array, double val)
{
	for (int i = 0; i < img_array->m; ++i)
	{
		for (int j = 0; j < img_array->n; ++j)
		{
			img_array->pixels[i][j] = val;
		}
	}
}
