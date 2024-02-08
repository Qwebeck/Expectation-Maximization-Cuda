// https://github.com/juliennonin/variational-gaussian-mixture
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mixture_models.h"
#include "stats.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include "linalg/eigen.h"

std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(0.0, 1.0);

void EM(matrix *data, int n_components, matrix **mixture_means, matrix ***mixture_covs, vector **mixture_weights)
{
    matrix *means = *mixture_means == NULL ? initialize_means(n_components, data->n_col) : *mixture_means;
    matrix **covs = *mixture_covs == NULL ? initialize_covs(n_components, data->n_col) : *mixture_covs;
    vector *weights = *mixture_weights == NULL ? initialize_weights(n_components) : *mixture_weights;

    // expectation step
    matrix *responsibilities = matrix_zeros(data->n_row, n_components);

    for (int weight_idx = 0; weight_idx < responsibilities->n_col; weight_idx++)
    {
        vector *probabilities = pdf(matrix_copy(data), matrix_row_copy(means, weight_idx), covs[weight_idx]);
        for (int i = 0; i < data->n_row; i++)
        {
            MATRIX_IDX_INTO(responsibilities, i, weight_idx) = VECTOR_IDX_INTO(weights, weight_idx) * VECTOR_IDX_INTO(probabilities, i);
        }
    }

    for (int i = 0; i < data->n_row; i++)
    {
        double sum = vector_sum(matrix_row_view(responsibilities, i));
        for (int j = 0; j < n_components; j++)
        {
            MATRIX_IDX_INTO(responsibilities, i, j) /= sum;
        }
    }

    // maximization step
    vector *sum_responsibilities = vector_zeros(responsibilities->n_col);
    for (int j = 0; j < sum_responsibilities->length; j++)
    {
        VECTOR_IDX_INTO(sum_responsibilities, j) = vector_sum(matrix_column_copy(responsibilities, j));
    }

    for (int i = 0; i < weights->length; i++)
    {
        VECTOR_IDX_INTO(weights, i) = VECTOR_IDX_INTO(sum_responsibilities, i) / data->n_row;
    }

    means = matrix_multiply(matrix_transpose(responsibilities), data);
    for (int i = 0; i < means->n_row; i++)
    {
        for (int j = 0; j < means->n_col; j++)
        {
            MATRIX_IDX_INTO(means, i, j) /= VECTOR_IDX_INTO(sum_responsibilities, i);
        }
    }
    *mixture_means = means;

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
    *mixture_covs = covs;
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
