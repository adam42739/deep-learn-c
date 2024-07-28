#pragma once
#include <stdlib.h>
#include <math.h>

double _rand_between(double lower, double upper);

typedef int __rng_dist_type;

#define RNG_XAVIER 0
#define RNG_HE 1

#define PI 3.14159265358979323846

double _erf_inv(double x);

double _rng_normal(double mu, double sigma);

double _rng_dist(__rng_dist_type rng_type, int n);

double _rng_xavier(int n);

double _rng_he(int n);
