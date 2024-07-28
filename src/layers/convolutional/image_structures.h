#pragma once
#include "../../general_lib/stdlib_mod.h"

typedef struct ImageArray
{
    double *pixels;
    int m;
    int n;
} ImageArray;

ImageArray* img_array_alloc(int m, int n);

void img_array_free(ImageArray* img_array);

typedef struct ImageLayer
{
    ImageArray **images;
    int num_images;
    int m;
    int n;
} ImageLayer;

ImageLayer* img_layer_alloc(int num_images, int m, int n);

void img_layer_free(ImageLayer* img_layer);
