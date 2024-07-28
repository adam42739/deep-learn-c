#include "convolutional_filter.h"

ConvolutionalFilter *cnn_filter_alloc(int m, int n)
{
    ConvolutionalFilter *cnn_filter = _mem_alloc(sizeof(ConvolutionalFilter));
    cnn_filter->linmod = linmod_alloc(m * n);
    cnn_filter->m = m;
    cnn_filter->n = n;
    return cnn_filter;
}

void cnn_filter_free(ConvolutionalFilter *cnn_filter)
{
    linmod_free(cnn_filter->linmod);
    free(cnn_filter);
}

void cnn_filter_randomize_weights(ConvolutionalFilter *cnn_filter, __rng_dist_type rng_type)
{
    linmod_randomize_weights(cnn_filter->linmod, rng_type);
}

void cnn_filter_set_bias_zero(ConvolutionalFilter *cnn_filter)
{
    linmod_set_bias_zero(cnn_filter->linmod);
}
