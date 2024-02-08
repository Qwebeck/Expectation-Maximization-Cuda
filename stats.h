#ifndef STATS_H
#define STATS_H
#include "linalg/eigen.h"

struct PSD
{
    matrix *U;
    double log_pdet;
    int rank;
};

vector *pdf(matrix *x, vector *mean, matrix *covariance);
PSD calculate_positive_semidefinite_matrix(matrix *data, int n_cols, int n_rows);
vector *pinv_1d(vector *v, double eps);
vector *calculate_log_pdf(matrix *x, vector *mean, matrix *U, double log_pdet, int rank);

#endif