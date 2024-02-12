#ifndef MIXTURE_MODELS_H
#define MIXTURE_MODELS_H
#include "../linalg/eigen.h"
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-6
#define LOG_2PI 1.8378770664093453
#define TOL 1e-6
#define MAX_ITER 10000

// Initializes an empty array filled with values
void EM(matrix *data, int n_components, matrix **mixture_means = NULL, matrix ***mixture_covs = NULL, vector **mixture_weights = NULL);

void normalize_responsibilities(matrix *data, matrix *&responsibilities, int n_components);

void estimate_covariance(int n_components, matrix *&data, matrix *&means, matrix *&responsibilities, vector *&sum_responsibilities, matrix **covs);

void estimate_means(matrix *&means, matrix *responsibilities, matrix *data, vector *&sum_responsibilities);

void sum_responsibilities(vector *&sum_responsibilities, int &j, matrix *responsibilities);

void estimate_log_responsibility(matrix *&responsibilities, int &i, int &weight_idx, vector *&weights, vector *&probabilities);

matrix *initialize_means(int n_components, int n_col);
matrix **initialize_covs(int n_components, int n_col);
vector *initialize_weights(int n_components);

#endif // MIXTURE_MODELS_H