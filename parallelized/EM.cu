// https://github.com/juliennonin/variational-gaussian-mixture
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "EM.h"
#include "stats.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include "../linalg/eigen.h"

#define IDX(i, j, n_cols) ((i) * (n_cols) + (j))

std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(0.0, 1.0);

void EM(matrix *data, int n_components, matrix **mixture_means, matrix ***mixture_covs, vector **mixture_weights)
{
    matrix *means = *mixture_means == NULL ? initialize_means(n_components, data->n_col) : *mixture_means;
    matrix **covs = *mixture_covs == NULL ? initialize_covs(n_components, data->n_col) : *mixture_covs;
    vector *weights = *mixture_weights == NULL ? initialize_weights(n_components) : *mixture_weights;

    // expectation step
    // double *d_data;
    // cudaMalloc((void **)&d_data, data->n_row * data->n_col * sizeof(double));
    // cudaMemcpy(d_data, DATA(data), data->n_row * data->n_col * sizeof(double), cudaMemcpyHostToDevice);

    matrix *responsibilities = matrix_zeros(data->n_row, n_components);

    double *d_responsibilities;
    cudaMalloc((void **)&d_responsibilities, responsibilities->n_row * responsibilities->n_col * sizeof(double));
    cudaMemcpy(d_responsibilities, DATA(responsibilities), responsibilities->n_row * responsibilities->n_col * sizeof(double), cudaMemcpyHostToDevice);

    for (int weight_idx = 0; weight_idx < responsibilities->n_col; weight_idx++)
    {
        /* 3. Estimation of log Gaussian probability: GAUSS */
        /* 4. Computation of weighted log probability: WLP */
        vector *probabilities = pdf(matrix_copy(data), matrix_row_copy(means, weight_idx), covs[weight_idx]);
        double *d_probabilities;
        cudaMalloc((void **)&d_probabilities, probabilities->length * sizeof(double));

        /* 5. Estimation of log responsibility: LR*/

        double weight = VECTOR_IDX_INTO(weights, weight_idx);

        // for (int i = 0; i < data->n_row; i++)
        // {
        dim3 block_size(BLOCK_SIZE);
        dim3 grid_size((data->n_row) / block_size.x + 1);

        cudaMemcpy(d_probabilities, DATA(probabilities), probabilities->length * sizeof(double), cudaMemcpyHostToDevice);

        estimate_log_responsibility<<<block_size, grid_size>>>(d_responsibilities, responsibilities->n_col, responsibilities->n_row, weight_idx, weight, d_probabilities, probabilities->length);
        // }
        cudaFree(d_probabilities);
    }

    cudaMemcpy(DATA(responsibilities), d_responsibilities, responsibilities->n_row * responsibilities->n_col * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_responsibilities);

    normalize_responsibilities(data, responsibilities, n_components);
    // maximization step

    /* 6. Computation of Sum of Resp: SR */
    vector *sum_responsibilities = vector_zeros(responsibilities->n_col);
    for (int j = 0; j < sum_responsibilities->length; j++)
    {
        sum_up_responsibilities(sum_responsibilities, j, responsibilities);
    }

    for (int i = 0; i < weights->length; i++)
    {
        VECTOR_IDX_INTO(weights, i) = VECTOR_IDX_INTO(sum_responsibilities, i) / data->n_row;
    }

    /* 8. Estimation of Mean of each cluster: MEC */
    estimate_means(means, responsibilities, data, sum_responsibilities);
    *mixture_means = means;

    /* 9. Estimation of Covariance of Each Cluster: CEC */
    estimate_covariance(n_components, data, means, responsibilities, sum_responsibilities, covs);
    *mixture_covs = covs;
}

// template <unsigned int blockSize>
void normalize_responsibilities(matrix *data, matrix *responsibilities, int n_components)
{
    for (int i = 0; i < data->n_row; i++)
    {
        double sum = vector_sum(matrix_row_view(responsibilities, i));

        for (int j = 0; j < n_components; j++)
        {
            MATRIX_IDX_INTO(responsibilities, i, j) /= sum;
        }
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}
template <unsigned int blockSize>
__global__ void reduce_row(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce(sdata, tid);
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

void estimate_covariance(int n_components, matrix *&data, matrix *&means, matrix *&responsibilities, vector *&sum_responsibilities, matrix **covs)
{
    for (int component = 0; component < n_components; component++)
    {
        matrix *j = matrix_zeros(data->n_row, data->n_col);
        for (int k = 0; k < data->n_row; k++)
        {
            for (int l = 0; l < data->n_col; l++)
            {
                MATRIX_IDX_INTO(j, k, l) = MATRIX_IDX_INTO(data, k, l) - MATRIX_IDX_INTO(means, component, l);
            }
        }

        matrix **s = init_array(data->n_row, matrix_zeros(data->n_col, data->n_col));
        // calculating dot product which will go to s. Test it, but almost sure
        for (int x = 0; x < data->n_row; ++x)
        {
            for (int y = 0; y < j->n_col; ++y)
            {
                for (int z = 0; z < j->n_col; ++z)
                {
                    MATRIX_IDX_INTO(s[x], y, z) = MATRIX_IDX_INTO(j, x, y) * MATRIX_IDX_INTO(j, x, z);
                }
            }
        }

        // Almost sure
        matrix *sigma = matrix_zeros(data->n_col, data->n_col);

        for (int i = 0; i < sigma->n_row; ++i)
        {
            for (int j = 0; j < sigma->n_col; ++j)
            {
                double cell_value = 0;
                for (int row_idx = 0; row_idx < data->n_row; ++row_idx)
                {
                    cell_value += MATRIX_IDX_INTO(responsibilities, row_idx, component) * MATRIX_IDX_INTO(s[row_idx], i, j);
                }
                cell_value /= VECTOR_IDX_INTO(sum_responsibilities, component);
                MATRIX_IDX_INTO(sigma, i, j) = cell_value;
            }
        }
        covs[component] = sigma;
    }
}

void estimate_means(matrix *&means, matrix *responsibilities, matrix *data, vector *&sum_responsibilities)
{
    means = matrix_multiply(matrix_transpose(responsibilities), data);
    for (int i = 0; i < means->n_row; i++)
    {
        for (int j = 0; j < means->n_col; j++)
        {
            MATRIX_IDX_INTO(means, i, j) /= VECTOR_IDX_INTO(sum_responsibilities, i);
        }
    }
}

void sum_up_responsibilities(vector *&sum_responsibilities, int &j, matrix *responsibilities)
{
    VECTOR_IDX_INTO(sum_responsibilities, j) = vector_sum(matrix_column_copy(responsibilities, j));
}

__global__ void estimate_log_responsibility(double *responsibilities, int resp_col, int resp_row, int weight_idx, double weight, double *probabilities, int p_length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= resp_row || i >= p_length)
    {
        return;
    }
    // MATRIX_IDX_INTO(responsibilities, i, weight_idx) = VECTOR_IDX_INTO(weights, weight_idx) * VECTOR_IDX_INTO(probabilities, i);
    responsibilities[i * resp_col + weight_idx] = weight * probabilities[i]; // uncoalesced access
}

matrix *initialize_means(int n_components, int n_col)
{
    matrix *means = matrix_zeros(n_components, n_col);
    for (int i = 0; i < n_components; i++)
    {
        for (int j = 0; j < means->n_col; j++)
        {
            MATRIX_IDX_INTO(means, i, j) = dis(gen);
        }
    }
    return means;
}

matrix **initialize_covs(int n_components, int n_col)
{
    matrix **covs = init_array(n_components, matrix_zeros(n_col, n_col));
    for (int i = 0; i < n_components; i++)
    {
        for (int j = 0; j < n_col; j++)
        {
            MATRIX_IDX_INTO(covs[i], j, j) = 1.0;
        }
    }
    return covs;
}

vector *initialize_weights(int n_components)
{
    vector *weights = vector_new(n_components);
    for (int i = 0; i < n_components; i++)
    {
        VECTOR_IDX_INTO(weights, i) = 1.0 / n_components;
    }
    return weights;
}
