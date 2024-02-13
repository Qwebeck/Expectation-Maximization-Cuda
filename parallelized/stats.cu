#include "stats.h"
#include <math.h>
#include "../linalg/eigen.h"
#include "EM.h"
#include <cblas.h>
#include "errorChecking.cu"
#include <iostream>

#define TILE_DIM 32
__global__ void gemm_tiled(double *A, double *B, double *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
{

    double CValue = 0;

    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ double As[TILE_DIM][TILE_DIM];
    __shared__ double Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++)
    {

        if (k * TILE_DIM + threadIdx.x < ACols && Row < ARows)
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols)
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}

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

vector *calculate_log_pdf(matrix *x, vector *mean, matrix *U, double log_pdet, int rank)
{
    matrix *dev = matrix_new(x->n_row, x->n_col);
    for (int i = 0; i < dev->n_row; ++i)
    {
        for (int j = 0; j < dev->n_col; j++)
        {
            MATRIX_IDX_INTO(dev, i, j) = MATRIX_IDX_INTO(x, i, j) - VECTOR_IDX_INTO(mean, j);
        }
    }

    double *d_dev;
    cudaMalloc(&d_dev, dev->n_row * dev->n_col * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dev, DATA(dev), dev->n_row * dev->n_col * sizeof(double), cudaMemcpyHostToDevice));

    double *d_U;
    cudaMalloc(&d_U, U->n_row * U->n_col * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(d_U, DATA(U), U->n_row * U->n_col * sizeof(double), cudaMemcpyHostToDevice));

    matrix *mahalanobis_distance = matrix_zeros(dev->n_row, U->n_col);
    double *d_mahalanobis_distance;
    CHECK_CUDA_ERROR(cudaMalloc(&d_mahalanobis_distance, mahalanobis_distance->n_row * mahalanobis_distance->n_col * sizeof(double)));
    cudaMemcpy(d_mahalanobis_distance, DATA(mahalanobis_distance), mahalanobis_distance->n_row * mahalanobis_distance->n_col * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((mahalanobis_distance->n_col + dimBlock.x - 1) / dimBlock.x, (mahalanobis_distance->n_row + dimBlock.y - 1) / dimBlock.y);
    gemm_tiled<<<dimGrid, dimBlock>>>(d_dev, d_U, d_mahalanobis_distance, dev->n_row, dev->n_col, U->n_row, U->n_col, mahalanobis_distance->n_row, mahalanobis_distance->n_col);

    CHECK_CUDA_ERROR(cudaMemcpy(DATA(mahalanobis_distance), d_mahalanobis_distance, mahalanobis_distance->n_row * mahalanobis_distance->n_col * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_dev);
    cudaFree(d_U);
    cudaFree(d_mahalanobis_distance);

    for (int i = 0; i < mahalanobis_distance->n_row; i++)
    {
        for (int j = 0; j < mahalanobis_distance->n_col; j++)
        {
            double sq = pow(MATRIX_IDX_INTO(mahalanobis_distance, i, j), 2);
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
