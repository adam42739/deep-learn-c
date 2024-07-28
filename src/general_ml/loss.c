#include "loss.h"

double _loss_function(double *output, double *expected, int n, __loss_type loss_type)
{
    switch (loss_type)
    {
    case LOSS_MSE:
        return _loss_mse(output, expected, n);
    default:
        return 0;
    }
}

double _loss_mse(double *output, double *expected, int n)
{
    double loss = 0;
    for (int i = 0; i < n; ++i)
    {
        loss += pow(output[i] - expected[i], 2);
    }
    return loss / n;
}

void _loss_deriv(double *output, double *expected, double *deriv, int n, __loss_type loss_type)
{
    switch (loss_type)
    {
    case LOSS_MSE:
        _loss_mse_deriv(output, expected, deriv, n);
        break;
    default:
        break;
    }
}

void _loss_mse_deriv(double *output, double *expected, double *deriv, int n)
{
    for (int i = 0; i < n; ++i)
    {
        deriv[i] = 2 * (output[i] - expected[i]) / n;
    }
}
