#ifndef MIXTURE_MODELS_H
#define MIXTURE_MODELS_H

#include <stdio.h>
#include <stdlib.h>
struct Data
{
    double *data;
    int num_records;
    int D;
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
double *init_array(int size, double value);
double *matrix_multiply(double *matrix1, double *matrix2, int row_count1, int col_count1, int row_count2, int col_count2);
double *calculate_self_dot_product(double *matrix, int row_count, int col_count);
double *invert_matrix(double *original, int dimension);
double sum(double *array, int size);
double pdf(vector *x, double *means, double *sigma, int dimension);
void EM(matrix *data, int n_components, double **mixture_means = NULL, double **mixture_covs = NULL, double **mixture_weights = NULL);
void store_csv(const char *filename, double *data, int row_count, int D);
double calculate_determinant(double *matrix, int dimension);
double *calculate_cofactor(double *original, int dimension);

#endif // MIXTURE_MODELS_H