#include "image_structures.h"

ImageArray* img_array_alloc(int m, int n)
{
	ImageArray* img_array = _mem_alloc(sizeof(ImageArray));
	img_array->pixels = _mem_alloc(sizeof(double) * m * n);
	img_array->m = m;
	img_array->n = n;
	return img_array;
}

void img_array_free(ImageArray* img_array)
{
	free(img_array->pixels);
	free(img_array);
}

ImageLayer* img_layer_alloc(int num_images, int m, int n)
{
	ImageLayer* img_layer = _mem_alloc(sizeof(ImageLayer));
	img_layer->images = _mem_alloc(sizeof(ImageArray*) * num_images);
	for (int i = 0; i < num_images; ++i)
	{
		img_layer->images[i] = img_array_alloc(m, n);
	}
	img_layer->num_images = num_images;
	img_layer->m = m;
	img_layer->n = n;
	return img_layer;
}

void img_layer_free(ImageLayer* img_layer)
{
	for (int i = 0; i < img_layer->num_images; ++i)
	{
		img_array_free(img_layer->images[i]);
	}
	free(img_layer->images);
	free(img_layer);
}

