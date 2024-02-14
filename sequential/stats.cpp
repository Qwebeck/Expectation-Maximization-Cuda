#include "stats.h"
#include <math.h>
#include "../linalg/eigen.h"
#include "EM.h"

/**
 * Calculates probability density function for each row of matrix x given mean and covariance
 * @param x - matrix of data points, shape (n_samples, n_features)
 * @param mean - mean of the distribution, shape (n_features,)
 * @param covariance - covariance matrix, shape (n_features, n_features)
 */
vector *pdf(matrix *x, vector *mean, matrix *covariance)
{
    // TODO: work on positive semidefinite matrix
    PSD psd = calculate_positive_semidefinite_matrix(covariance, covariance->n_col, covariance->n_row);
    vector *log_pdf = calculate_log_pdf(x, mean, psd.U, psd.log_pdet, psd.rank);
    vector *result = vector_new(log_pdf->length);
    for (int i = 0; i < log_pdf->length; i++)
    {
        VECTOR_IDX_INTO(result, i) = exp(VECTOR_IDX_INTO(log_pdf, i));
    }
    return result;
}

/**
 * log_pdet - logarithm of determinant of pseudoinverse matrix
 * rank - rank of matrix
 */

PSD calculate_positive_semidefinite_matrix(matrix *data, int n_cols, int n_rows)
{
    eigen *eigen = eigen_solve(data, TOL, MAX_ITER);
    vector *s = vector_new(n_cols);
    vector *s_pinv = pinv_1d(eigen->eigenvalues, TOL);
    for (int i = 0; i < n_cols; i++)
    {
        VECTOR_IDX_INTO(s_pinv, i) = sqrt(VECTOR_IDX_INTO(s_pinv, i));
    }

    matrix *U = matrix_new(eigen->eigenvectors->n_row, eigen->eigenvectors->n_col);
    for (int i = 0; i < U->n_row; i++)
    {
        for (int j = 0; j < U->n_col; j++)
        {
            MATRIX_IDX_INTO(U, i, j) = MATRIX_IDX_INTO(eigen->eigenvectors, i, j) * VECTOR_IDX_INTO(s_pinv, j);
        }
    }
    float log_pdet = 0;
    for (int i = 0; i < n_cols; i++)
    {
        log_pdet += log(VECTOR_IDX_INTO(eigen->eigenvalues, i));
    }

    return {
        .U = U,
        .log_pdet = log_pdet,
        .rank = n_cols};
}

vector *pinv_1d(vector *v, float eps)
{
    // float *result = (float *)init_array(size, 0); // fix
    vector *result = vector_zeros(v->length);
    for (int i = 0; i < v->length; i++)
    {
        if (abs(VECTOR_IDX_INTO(v, i)) > eps)
        {
            VECTOR_IDX_INTO(result, i) = 1 / VECTOR_IDX_INTO(v, i);
        }
    }
    return result;
}

vector *calculate_log_pdf(matrix *x, vector *mean, matrix *U, float log_pdet, int rank)
{
    matrix *dev = matrix_new(x->n_row, x->n_col);
    for (int i = 0; i < dev->n_row; ++i)
    {
        for (int j = 0; j < dev->n_col; j++)
        {
            MATRIX_IDX_INTO(dev, i, j) = MATRIX_IDX_INTO(x, i, j) - VECTOR_IDX_INTO(mean, j);
        }
    }

    matrix *mahalanobis_distance = matrix_multiply(dev, U);
    for (int i = 0; i < mahalanobis_distance->n_row; i++)
    {
        for (int j = 0; j < mahalanobis_distance->n_col; j++)
        {
            float sq = pow(MATRIX_IDX_INTO(mahalanobis_distance, i, j), 2);
            MATRIX_IDX_INTO(mahalanobis_distance, i, j) = sq;
        }
    }
    vector *result = vector_zeros(dev->n_row);
    for (int i = 0; i < result->length; i++)
    {
        for (int j = 0; j < mahalanobis_distance->n_col; j++)
        {
            VECTOR_IDX_INTO(result, i) += MATRIX_IDX_INTO(mahalanobis_distance, i, j);
        }
        VECTOR_IDX_INTO(result, i) = -0.5 * (rank * LOG_2PI + log_pdet + VECTOR_IDX_INTO(result, i));
    }
    return result;
}
