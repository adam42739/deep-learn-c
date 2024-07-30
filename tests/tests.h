#pragma once
#include "../src/vanilla_sgd.h"
#include "../src/linear_network.h"
#include "../src/general_ml/loss.h"
#include "../src/general_lib/random.h"
#include "../src/layers/convolutional/convolutional_layer.h"
#include <stdio.h>

void test_linear_sgd(void);

void test_cnn_forward(void);

void test_cnn_alloc_dealloc(void);
