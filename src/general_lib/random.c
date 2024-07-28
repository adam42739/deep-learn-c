#include "random.h"

double _rand_between(double lower, double upper)
{
	int rng = rand();
	double between01 = (rng + 0.5) / (RAND_MAX + 1.0);
	return lower + (upper - lower) * between01;
}

double _erf_inv(double x)
{
	double tt1, tt2, lnx, sgn;
	sgn = (x < 0) ? -1.0f : 1.0f;
	x = (1 - x) * (1 + x);
	lnx = log(x);
	tt1 = 2 / (PI * 0.147) + 0.5f * lnx;
	tt2 = 1 / (0.147) * lnx;
	return (sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)));
}

double _rng_normal(double mu, double sigma)
{
	double x = 2 * _rand_between(0, 1) - 1;
	double erf = _erf_inv(x);
	return mu + sigma * sqrt(2) * erf;
}

double _rng_dist(__rng_dist_type rng_type, int n)
{
	switch (rng_type)
	{
	case RNG_XAVIER:
		return _rng_xavier(n);
	case RNG_HE:
		return _rng_he(n);
	default:
		return 0;
	}
}

double _rng_xavier(int n)
{
	return _rng_normal(0, sqrt(1.0 / n));
}

double _rng_he(int n)
{
	return _rng_normal(0, sqrt(2.0 / n));
}
