#pragma once
#include "../../general_lib/stdlib_mod.h"

typedef struct ImageArray
{
	double** pixels;
	int m;
	int n;
} ImageArray;

ImageArray* img_array_alloc(int m, int n);

void img_array_free(ImageArray* img_array);

void img_array_sum(ImageArray** img_array, int num_array, ImageArray* output);

void img_array_product(ImageArray* img_array_A, ImageArray* img_array_B, ImageArray* output);

void img_array_copy(ImageArray* source, ImageArray* dest);

typedef struct ImageLayer
{
	ImageArray** img_arrays;
	int num_arrays;
	int m;
	int n;
} ImageLayer;

ImageLayer* img_layer_alloc(int num_arrays, int m, int n);

void img_layer_free(ImageLayer* img_layer);
