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
