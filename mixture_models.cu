// https://github.com/juliennonin/variational-gaussian-mixture
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mixture_models.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include "linalg/eigen.h"

/*
TODO:
3. implement c sequential code [x]
4. debug c sequential code [ ]


- verify how to compute probability density function
--------------
pdf
        dim, mean, cov = self._process_parameters(None, mean, cov)
        x = self._process_quantiles(x, dim)
        psd = _PSD(cov, allow_singular=allow_singular)
        out = np.exp(self._logpdf(x, mean, psd.U, psd.log_pdet, psd.rank))
        return _squeeze_output(out)
---------------
where _PSD has the following code:


    s, u = scipy.linalg.eigh(M, lower=lower, check_finite=check_finite)

    eps = _eigvalsh_to_eps(s, cond, rcond)
    if np.min(s) < -eps:
        msg = "The input matrix must be symmetric positive semidefinite."
        raise ValueError(msg)
    d = s[s > eps]
    if len(d) < len(s) and not allow_singular:
        msg = ("When `allow_singular is False`, the input matrix must be "
               "symmetric positive definite.")
        raise np.linalg.LinAlgError(msg)
    s_pinv = _pinv_1d(s, eps)
    U = np.multiply(u, np.sqrt(s_pinv))
    # Initialize the eagerly precomputed attributes.
    self.rank = len(d)
    self.U = U
    self.log_pdet = np.sum(np.log(d))

and log_pdf:
        dev = x - mean
        maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
        return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)
-----
I need to:
1. calculate eigenvalues decomposition to get U and eigenvalues from it
2. sum logarithms of eigenvalues
3. calculate rank
4. calculate log_pdf

TODO tomorrow:
reimplement this https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html



- write and test code for 3D matrix (which is covariance matrix)

--- can we use cublas?
*/

#define TOL 1e-6
#define MAX_ITER 10000

int main(int argc, char **argv)
{
    char *filename = argv[1];
    int n_components = atoi(argv[2]);
    int row_count = atoi(argv[3]);
    int D = atoi(argv[4]);
    char *means_fname = argv[5];
    char *covs_fname = argv[6];
    int max_iter = atoi(argv[7]);
    double *csv_contet = read_csv(filename, row_count, D);

    matrix *data = matrix_from_array(csv_contet, row_count, D);

    matrix *means;
    matrix **covs;
    vector *weights;
    for (int i = 0; i < max_iter; i++)
    {
        EM(data, n_components, &means, &covs, &weights);
    }

    double means_array[n_components * D];
    for (int i = 0; i < n_components; i++)
    {
        for (int j = 0; j < D; j++)
        {
            means_array[i * D + j] = MATRIX_IDX_INTO(means, i, j);
        }
    }
    store_csv(means_fname, means_array, n_components, D);
    double covs_array[n_components * D * D];
    for (int i = 0; i < n_components; i++)
    {
        for (int j = 0; j < D; j++)
        {
            for (int k = 0; k < D; k++)
            {
                covs_array[i * D * D + j * D + k] = MATRIX_IDX_INTO(covs[i], j, k);
            }
        }
    }
    store_csv(covs_fname, covs_array, n_components, D * D);

    return 0;
}

void EM(matrix *data, int n_components, matrix **mixture_means, matrix ***mixture_covs, vector **mixture_weights)
{
    matrix *means = mixture_means == NULL ? NULL : *mixture_means;

    if (means == NULL)
    {
        means = matrix_new(n_components, data->n_col);
        for (int i = 0; i < n_components; i++)
        {
            for (int j = 0; j < data->n_col; j++)
            {
                MATRIX_IDX_INTO(means, i, j) = MATRIX_IDX_INTO(data, i, j);
            }
        }
        mixture_means = &means;
    }
    matrix **covs = mixture_covs == NULL ? NULL : *mixture_covs;
    if (covs == NULL)
    {
        covs = init_array(n_components, matrix_zeros(data->n_col, data->n_col));
        for (int i = 0; i < n_components; i++)
        {
            for (int j = 0; j < data->n_col; j++)
            {
                MATRIX_IDX_INTO(covs[i], j, j) = 1.0;
            }
        }
        mixture_covs = &covs;
    }

    vector *weights = mixture_weights == NULL ? NULL : *mixture_weights;
    if (weights == NULL)
    {
        weights = vector_new(n_components);
        for (int i = 0; i < n_components; i++)
        {
            VECTOR_IDX_INTO(weights, i) = 1.0 / n_components;
        }
        mixture_weights = &weights;
    }

    // expectation step
    matrix *responsibilities = matrix_zeros(data->n_row, n_components);

    for (int weight_idx = 0; weight_idx < n_components; weight_idx++)
    {
        vector *probabilities = pdf(data, matrix_column_copy(means, weight_idx), covs[weight_idx]);
        for (int i = 0; i < data->n_row; i++)
        {
            MATRIX_IDX_INTO(responsibilities, i, weight_idx) = VECTOR_IDX_INTO(weights, weight_idx) * VECTOR_IDX_INTO(probabilities, i);
        }
    }

    for (int i = 0; i < data->n_row; i++)
    {
        double sum = 0;
        for (int j = 0; j < n_components; j++)
        {
            sum += VECTOR_IDX_INTO(responsibilities, i * n_components + j);
        }
        for (int j = 0; j < n_components; j++)
        {
            MATRIX_IDX_INTO(responsibilities, n_components, j) /= sum;
        }
    }

    // maximization step
    vector *sum_responsibilities = vector_zeros(n_components);
    for (int j = 0; j < n_components; j++)
    {
        for (int i = 0; i < data->n_row; i++)
        {
            VECTOR_IDX_INTO(sum_responsibilities, j) += MATRIX_IDX_INTO(responsibilities, i, j) / data->n_row;
        }
    }

    for (int i = 0; i < weights->length; i++)
    {
        VECTOR_IDX_INTO(weights, i) = VECTOR_IDX_INTO(sum_responsibilities, i) / data->n_row;
    }

    means = matrix_multiply(matrix_transpose(responsibilities), data);
    for (int i = 0; i < data->n_row; i++)
    {
        for (int j = 0; j < data->n_col; j++)
        {
            MATRIX_IDX_INTO(means, i, j) /= VECTOR_IDX_INTO(sum_responsibilities, i);
        }
    }
    mixture_means = &means;

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
        for (int row_idx = 0; row_idx < data->n_row; row_idx++)
        {
            for (int kk = 0; kk < data->n_col; kk++)
            {
                double cell_value = 0;
                for (int l = 0; l < data->n_col; l++)
                {
                    cell_value += MATRIX_IDX_INTO(j, row_idx, l) * MATRIX_IDX_INTO(j, row_idx, l);
                }
                for (int kkk = 0; kkk < data->n_col; kkk++)
                {
                    MATRIX_IDX_INTO(s[row_idx], kk, kkk) = cell_value;
                }
            }
        }
        // Almost sure
        matrix *sigma = matrix_zeros(data->n_col, data->n_col);
        for (int i = 0; i < data->n_col; ++i)
        {
            for (int j = 0; j < data->n_col; ++j)
            {
                double cell_value = 0;
                for (int row_idx = 0; row_idx < data->n_row; ++row_idx)
                {
                    cell_value += MATRIX_IDX_INTO(responsibilities, row_idx, component) * MATRIX_IDX_INTO(s[i], j, row_idx);
                }
                cell_value /= VECTOR_IDX_INTO(sum_responsibilities, component);
                MATRIX_IDX_INTO(sigma, i, j) = cell_value;
            }
        }
        covs[component] = sigma;
    }
    mixture_covs = &covs;
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
    matrix *log_pdf = calculate_log_pdf(x, mean, psd.U, psd.log_pdet, psd.rank);
    vector *result = vector_new(log_pdf->n_row);
    for (int i = 0; i < log_pdf->n_row; i++)
    {
        VECTOR_IDX_INTO(result, i) = exp(MATRIX_IDX_INTO(log_pdf, i, 0));
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
    vector *result = vector_new(v->length);
    for (int i = 0; i < 1; i++)
    {
        if (abs(VECTOR_IDX_INTO(v, i)) > eps)
        {
            VECTOR_IDX_INTO(result, i) = 1 / VECTOR_IDX_INTO(v, i);
        }
    }
    return result;
}

matrix *calculate_log_pdf(matrix *x, vector *mean, matrix *U, double log_pdet, int rank)
{
    matrix *dev = matrix_new(x->n_row, x->n_col);
    for (int j = 0; j < dev->n_row; ++j)
    {
        for (int i = 0; i < dev->n_col; i++)
        {
            MATRIX_IDX_INTO(dev, j, i) = MATRIX_IDX_INTO(x, j, i) - VECTOR_IDX_INTO(mean, i);
        }
    }

    matrix *mahalanobis_distance = matrix_multiply(dev, U);
    for (int i = 0; i < mahalanobis_distance->n_row; i++)
    {
        for (int j = 0; j < mahalanobis_distance->n_col; j++)
        {
            double sq = pow(MATRIX_IDX_INTO(dev, i, j), 2);
            MATRIX_IDX_INTO(mahalanobis_distance, i, j) = sq;
        }
    }
    matrix *result = matrix_zeros(dev->n_row, 1);
    for (int i = 0; i < result->n_row; i++)
    {
        for (int j = 0; j < result->n_col; j++)
        {
            MATRIX_IDX_INTO(result, i, j) += MATRIX_IDX_INTO(mahalanobis_distance, i, j);
        }
        MATRIX_IDX_INTO(result, i, 0) = -0.5 * (rank * log(2 * M_PI) + log_pdet + MATRIX_IDX_INTO(result, i, 0));
    }
    return result;
}
