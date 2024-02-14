#include "stats.h"
#include <math.h>
#include "../linalg/eigen.h"
#include "EM.h"
#include <cblas.h>
#include "errorChecking.cu"
#include <iostream>
#include "matrix_operations.cuh"
#include "utils.h"

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
    double log_pdet = 0;
    for (int i = 0; i < n_cols; i++)
    {
        log_pdet += log(VECTOR_IDX_INTO(eigen->eigenvalues, i));
    }

    return {
        .U = U,
        .log_pdet = log_pdet,
        .rank = n_cols};
}

vector *pinv_1d(vector *v, double eps)
{
    // double *result = (double *)init_array(size, 0); // fix
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

__global__ void subtract_row_vector(double *dev, double *x, double *mean, int n_row, int n_col)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_row && j < n_col)
    {
        dev[i * n_col + j] = x[i * n_col + j] - mean[j];
    }
}

__global__ void log_probability_map(double *x, int n_col, int rank, double log_pdet)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_col)
    {
        x[idx] = -0.5 * (rank * LOG_2PI + log_pdet + x[idx]);
    }
}

vector *calculate_log_pdf(matrix *x, vector *mean, matrix *U, double log_pdet, int rank)
{
    matrix *dev = matrix_zeros(x->n_row, x->n_col);
    // for (int i = 0; i < dev->n_row; ++i)
    // {
    //     for (int j = 0; j < dev->n_col; j++)
    //     {
    //         MATRIX_IDX_INTO(dev, i, j) = MATRIX_IDX_INTO(x, i, j) - VECTOR_IDX_INTO(mean, j);
    //     }
    // }
    double *d_x;
    cudaMalloc(&d_x, x->n_row * x->n_col * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, DATA(x), x->n_row * x->n_col * sizeof(double), cudaMemcpyHostToDevice));

    double *d_dev;
    cudaMalloc(&d_dev, dev->n_row * dev->n_col * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dev, DATA(dev), dev->n_row * dev->n_col * sizeof(double), cudaMemcpyHostToDevice));

    double *d_mean;
    cudaMalloc(&d_mean, mean->length * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_mean, DATA(mean), mean->length * sizeof(double), cudaMemcpyHostToDevice));

    dim3 dimDevBlock(mean->length, TILE_DIM);
    dim3 gridSize((dev->n_col + dimDevBlock.x - 1) / dimDevBlock.x, (dev->n_row + dimDevBlock.y - 1) / dimDevBlock.y);
    subtract_row_vector<<<gridSize, dimDevBlock>>>(d_dev, d_x, d_mean, dev->n_row, dev->n_col);

    double *d_U;
    cudaMalloc(&d_U, U->n_row * U->n_col * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_U, DATA(U), U->n_row * U->n_col * sizeof(double), cudaMemcpyHostToDevice));

    matrix *mahalanobis_distance = matrix_zeros(dev->n_row, U->n_col);
    double *d_mahalanobis_distance;
    CHECK_CUDA_ERROR(cudaMalloc(&d_mahalanobis_distance, mahalanobis_distance->n_row * mahalanobis_distance->n_col * sizeof(double)));
    cudaMemcpy(d_mahalanobis_distance, DATA(mahalanobis_distance), mahalanobis_distance->n_row * mahalanobis_distance->n_col * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimMahalanobisBlock(TILE_DIM, TILE_DIM);
    dim3 dimMahalanobisGrid((mahalanobis_distance->n_col + dimMahalanobisBlock.x - 1) / dimMahalanobisBlock.x, (mahalanobis_distance->n_row + dimMahalanobisBlock.y - 1) / dimMahalanobisBlock.y);
    gemm_tiled<<<dimMahalanobisGrid, dimMahalanobisBlock>>>(d_dev, d_U, d_mahalanobis_distance, dev->n_row, dev->n_col, U->n_row, U->n_col, mahalanobis_distance->n_row, mahalanobis_distance->n_col);

    cudaFree(d_dev);
    cudaFree(d_U);

    elementwise_pow<<<dimMahalanobisGrid, dimMahalanobisBlock>>>(d_mahalanobis_distance, mahalanobis_distance->n_row, mahalanobis_distance->n_col, 2);

    // double *d_mahalanobis_distance_transposed;
    // matrix *mahalanobis_distance_transposed = matrix_zeros(mahalanobis_distance->n_col, mahalanobis_distance->n_row);
    // cudaMalloc(&d_mahalanobis_distance_transposed, mahalanobis_distance->n_row * mahalanobis_distance->n_col * sizeof(double));
    // CHECK_CUDA_ERROR(cudaMemcpy(d_mahalanobis_distance_transposed, DATA(mahalanobis_distance_transposed), mahalanobis_distance_transposed->n_row * mahalanobis_distance_transposed->n_col * sizeof(double), cudaMemcpyHostToDevice));

    // transpose<<<dimMahalanobisGrid, dimMahalanobisBlock>>>(d_mahalanobis_distance_transposed, d_mahalanobis_distance, mahalanobis_distance->n_col, mahalanobis_distance->n_row);

    vector *result = vector_zeros(dev->n_row);
    double *d_result;
    cudaMalloc(&d_result, result->length * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_result, DATA(result), result->length * sizeof(double), cudaMemcpyHostToDevice));

    reduce_matrix_rows<<<dimMahalanobisGrid, dimMahalanobisBlock>>>(d_mahalanobis_distance, mahalanobis_distance->n_row, mahalanobis_distance->n_col, d_result);

    dim3 resultBlock(TILE_DIM);
    dim3 resultGrid((result->length + resultBlock.x - 1) / resultBlock.x);
    log_probability_map<<<resultGrid, resultBlock>>>(d_result, result->length, rank, log_pdet);

    CHECK_CUDA_ERROR(cudaMemcpy(DATA(result), d_result, result->length * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_mahalanobis_distance);
    // cudaFree(d_mahalanobis_distance_transposed);
    cudaFree(d_result);
    // for (int i = 0; i < result->length; i++)
    // {
    //     for (int j = 0; j < mahalanobis_distance->n_col; j++)
    //     {
    //         VECTOR_IDX_INTO(result, i) += MATRIX_IDX_INTO(mahalanobis_distance, i, j);
    //     }
    //     VECTOR_IDX_INTO(result, i) = -0.5 * (rank * LOG_2PI + log_pdet + VECTOR_IDX_INTO(result, i));
    // }
    return result;
}
