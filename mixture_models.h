#ifndef MIXTURE_MODELS_H
#define MIXTURE_MODELS_H
#include "linalg/eigen.h"
#include <stdio.h>
#include <stdlib.h>
struct Data
{
    double *data;
    int num_records;
    int D;
};

struct PSD
{
    matrix *U;
    double log_pdet;
    int rank;
};

struct GmmHyperparams
{
    double alpha;
    double beta;
    double W;
    double nu;
    double m;
};
// Initializes an empty array filled with values
vector *calculate_log_pdf(matrix *x, vector *mean, matrix *U, double log_pdet, int rank);
vector *pdf(matrix *x, vector *mean, matrix *covariance);
void EM(matrix *data, int n_components, matrix **mixture_means = NULL, matrix ***mixture_covs = NULL, vector **mixture_weights = NULL);
vector *pinv_1d(vector *v, double eps);
PSD calculate_positive_semidefinite_matrix(matrix *data, int n_cols, int n_rows);

#endif // MIXTURE_MODELS_H