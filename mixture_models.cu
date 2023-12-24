// https://github.com/juliennonin/variational-gaussian-mixture
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpuDiagamma.c"
#include <math.h>
#include "mixture_models.h"
#include "utils.h"
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
#include <stdio.h>
#include <stdlib.h>
#include "linalg/eigen.h"

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

    matrix *matrix = matrix_from_array(csv_contet, row_count, D);

    double *means;
    double *covs;
    double *weights;
    for (int i = 0; i < max_iter; i++)
    {
        EM(matrix, n_components, &means, &covs, &weights);
    }
    store_csv(means_fname, means, n_components, D);
    store_csv(covs_fname, covs, n_components, D * D);

    return 0;
}

void EM(matrix *data, int n_components, double **mixture_means = NULL, double **mixture_covs = NULL, double **mixture_weights = NULL)
{
    double *means = mixture_means == NULL ? NULL : *mixture_means;

    if (means == NULL)
    {
        means = init_array(n_components * data->n_col, 1.0 / n_components);
        mixture_means = &means;
    }
    double *covs = mixture_covs == NULL ? NULL : *mixture_covs;
    if (covs == NULL)
    {
        covs = init_array(n_components * data->n_col * data->n_col, 0);
        for (int i = 0; i < n_components; i++)
        {
            for (int j = 0; j < data->n_col; j++)
            {
                covs[i * data->n_col * data->n_col + j * data->n_col + j] = 1.0;
            }
        }
        mixture_covs = &covs;
    }

    double *weights = mixture_weights == NULL ? NULL : *mixture_weights;
    if (weights == NULL)
    {
        weights = init_array(n_components, 1.0 / n_components);
        mixture_weights = &weights;
    }

    // expectation step
    double *responsibilities = init_array(data->n_row * n_components, 0);
    for (int j = 0; j < n_components; j++)
    {
        for (int i = 0; i < data->n_row; i++)
        {
            responsibilities[i * n_components + j] = weights[j] * pdf(matrix_row_view(data, i),
                                                                      &means[j * data->n_col], &covs[j * data->n_col * data->n_col], data->n_col);
        }

        for (int i = 0; i < data->n_row; i++)
        {
            responsibilities[i * n_components + j] /= sum(&responsibilities[i * n_components], n_components);
        }
    }

    // maximization step
    double *sum_responsibilities = init_array(n_components, 0);
    for (int j = 0; j < n_components; j++)
    {
        for (int i = 0; i < data->n_row; i++)
        {
            sum_responsibilities[j] += responsibilities[i * n_components + j] / data->n_row;
        }
    }

    means = matrix_multiply(responsibilities, sum_responsibilities, data->n_row, n_components, n_components, 1);

    for (int i = 0; i < n_components; ++i)
    {
        double *temp = init_array(data->n_row * data->n_col, 0);
        for (int i = 0; i < data->n_row; i++)
        {
            for (int j = 0; j < data->n_col; j++)
            {
                temp[i * data->n_col + j] = MATRIX_IDX_INTO(data, i, j);
            }
        }
        double *temp2 = init_array(data->n_row, 0);
        temp2 = calculate_self_dot_product(temp, data->n_row, data->n_col);
        // super unsure
        covs[i * data->n_col * data->n_col] = sum(matrix_multiply(responsibilities, temp2, data->n_row, n_components, n_components, 1), data->n_row) / sum_responsibilities[i];
    }
}

double *pinv_1d(double *numbers, double eps, int size)
{
    double *result = (double *)init_array(size, 0); // fix
    for (int i = 0; i < 1; i++)
    {
        if (abs(numbers[i]) > eps)
        {
            result[i] = 1 / numbers[i];
        }
    }
    return result;
}

void psd(double *data, int n_cols, int n_rows)
{
    matrix *m = matrix_from_array(data, n_rows, n_cols);
    eigen *eigen = eigen_solve(m, TOL, MAX_ITER);
    double *s = init_array(n_cols, 0);
    for (int i = 0; i < n_cols; i++)
    {
        s[i] = VECTOR_IDX_INTO(eigen->eigenvalues, i);
    }
    double *s_pinv = pinv_1d(s, TOL, n_cols);
    for (int i = 0; i < n_cols; i++)
    {
        s[i] = sqrt(s[i]);
    }
    matrix *S_ping = matrix_from_array(s_pinv, n_cols, 1);
    matrix *U = matrix_multiply(eigen->eigenvectors, S_ping);
    // log_pdget
}

void log_pdf(double *x, int n_rows, double mean, matrix *U, double log_pdet, int rank)
{
    matrix *deviation = matrix_from_array(x, 1, n_rows);
    matrix *tmp = matrix_multiply(deviation, U);
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_rows; j++)
        {
            double sq = sqrt(MATRIX_IDX_INTO(deviation, i, j));
            MATRIX_IDX_INTO(tmp, i, j) = sq;
        }
    }

    // matrix* maha = matrix_multiply(dev, U);
    // return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)) {
}

/**
 * Compute probability for vector x to belong to distribution. Generated, so pretty unsure about code.
 *
 *
 * ........
 */
double pdf(vector *x, double *means, double *sigma, int dimension)
{
    double det;
    double *temp = (double *)malloc(dimension * sizeof(double));
    for (int i = 0; i < dimension; ++i)
    {
        det *= sigma[i * dimension + i];
        temp[i] = VECTOR_IDX_INTO(x, i) - means[i];
    }

    double exp = 0.0;
    double *inverse = invert_matrix(sigma, dimension);
    for (int i = 0; i < dimension; ++i)
    {
        for (int j = 0; j < dimension; j++)
        {
            exp += temp[j] * inverse[j * dimension + i];
        }
        exp *= temp[i];
    }
    exp *= -0.5;

    return 1 / sqrt(pow(2 * M_PI, dimension) * det) * exp;
}

// Function to perform Gaussian elimination and invert a matrix
double *invert_matrix(double *original, int dimension)
{
    double determinant = calculate_determinant(original, dimension);
    if (determinant == 0.0)
    {
        // Matrix is singular (non-invertible)
        throw std::invalid_argument("received non invertible matrix");
    }

    double d = 1. / determinant;

    double *inverse = init_array(dimension * dimension, 0);

    double *cofactor = calculate_cofactor(original, dimension);

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            double sign = pow(-1, i + j);
            inverse[i * dimension + j] = d * sign * cofactor[i * dimension + j];
        }
    }
    inverse = transpose(inverse, dimension, dimension);
    return inverse;
}

// copilot generated
double calculate_determinant(double *matrix, int dimension)
{
    double det = 0;
    if (dimension == 1)
    {
        return matrix[0];
    }
    else if (dimension == 2)
    {
        return matrix[0] * matrix[3] - matrix[1] * matrix[2];
    }
    else
    {
        for (int i = 0; i < dimension; i++)
        {
            double *temp = init_array((dimension - 1) * (dimension - 1), 0);
            for (int j = 1; j < dimension; j++)
            {
                for (int k = 0; k < dimension; k++)
                {
                    if (k < i)
                    {
                        temp[(j - 1) * (dimension - 1) + k] = matrix[j * dimension + k];
                    }
                    else if (k > i)
                    {
                        temp[(j - 1) * (dimension - 1) + k - 1] = matrix[j * dimension + k];
                    }
                }
            }
            det += pow(-1, i) * matrix[i] * calculate_determinant(temp, dimension - 1);
        }
    }
    return det;
}

double *init_array(int size, double value)
{
    double *array = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++)
    {
        array[i] = value;
    }
    return array;
}

double sum(double *array, int size)
{
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum;
}

double *matrix_multiply(double *matrix1, double *matrix2, int row_count1, int col_count1, int row_count2, int col_count2)
{
    double *result = init_array(row_count1 * col_count2, 0);
    for (int i = 0; i < row_count1; i++)
    {
        for (int j = 0; j < col_count2; j++)
        {
            for (int k = 0; k < col_count1; k++)
            {
                result[i * col_count2 + j] += matrix1[i * col_count1 + k] * matrix2[k * col_count2 + j];
            }
        }
    }
    return result;
}

double *calculate_self_dot_product(double *matrix, int row_count, int col_count)
{
    double *result = init_array(row_count * col_count, 0);
    for (int i = 0; i < row_count; i++)
    {
        for (int j = 0; j < col_count; j++)
        {
            for (int k = 0; k < col_count; k++)
            {
                result[i * col_count + j] += matrix[i * col_count + k] * matrix[k * col_count + j];
            }
        }
    }
    return result;
}

double *transpose(double *matrix, int row_count, int col_count)
{
    double *result = init_array(row_count * col_count, 0);
    for (int i = 0; i < row_count; i++)
    {
        for (int j = 0; j < col_count; j++)
        {
            result[j * row_count + i] = matrix[i * col_count + j];
        }
    }
    return result;
}

// copilot generated
double *calculate_cofactor(double *original, int dimension)
{
    double *cofactor = init_array(dimension * dimension, 0);
    double *temp = init_array((dimension - 1) * (dimension - 1), 0);
    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            int sign = pow(-1, i + j);
            for (int k = 0; k < dimension; k++)
            {
                for (int l = 0; l < dimension; l++)
                {
                    if (k != i && l != j)
                    {
                        temp[(k < i ? k : k - 1) * (dimension - 1) + (l < j ? l : l - 1)] = original[k * dimension + l];
                    }
                }
            }
            cofactor[i * dimension + j] = sign * calculate_determinant(temp, dimension - 1);
        }
    }
    return cofactor;
}
