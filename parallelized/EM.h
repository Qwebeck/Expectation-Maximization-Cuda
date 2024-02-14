#ifndef EM_H
#define EM_H
#include "../linalg/eigen.h"
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-6
#define LOG_2PI 1.8378770664093453
#define TOL 1e-6
#define MAX_ITER 10000
#define BLOCK_SIZE 256

// Initializes an empty array filled with values
void EM(matrix *data, int n_components, matrix **mixture_means = NULL, matrix ***mixture_covs = NULL, vector **mixture_weights = NULL);

matrix *initialize_means(int n_components, int n_col);
matrix **initialize_covs(int n_components, int n_col);
vector *initialize_weights(int n_components);
void normalize_responsibilities(matrix *data, matrix *responsibilities, int n_components);
void estimate_covariance(int n_components, matrix *&data, matrix *&means, matrix *&responsibilities, vector *&sum_responsibilities, matrix **covs);
void estimate_means(matrix *&means, matrix *responsibilities, matrix *data, vector *&sum_responsibilities);
void sum_up_responsibilities(vector *&sum_responsibilities, int &j, matrix *responsibilities);
__global__ void estimate_log_responsibility(float *responsibilities, int resp_col, int resp_row, int weight_idx, float weight, float *probabilities, int p_length);

#endif // EM_H