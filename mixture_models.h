#ifndef MIXTURE_MODELS_H
#define MIXTURE_MODELS_H
#include "linalg/eigen.h"
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-6
#define LOG_2PI 1.8378770664093453
#define TOL 1e-6
#define MAX_ITER 10000

// Initializes an empty array filled with values
void EM(matrix *data, int n_components, matrix **mixture_means = NULL, matrix ***mixture_covs = NULL, vector **mixture_weights = NULL);

matrix *initialize_means(int n_components, int n_col);
matrix **initialize_covs(int n_components, int n_col);
vector *initialize_weights(int n_components);

#endif // MIXTURE_MODELS_H