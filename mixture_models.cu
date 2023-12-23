// https://github.com/juliennonin/variational-gaussian-mixture
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpuDiagamma.c"
#include <math.h>
#include "mixture_models.h"
/*
TODO:
3. implement c sequential code [x]
4. debug c sequential code [ ]


- verify how to compute probability density function
--------------
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

double *pinv_1d(float *numbers, float eps, int size)
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
    float *s = init_array(n_cols, 0);
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

void log_pdf(double *x, int n_rows, double mean, matrix *U, float log_pdet, int rank)
{
    matrix *dev = matrix_from_array(x, 1, n_rows);
    matrix *tmp = matrix_multiply(dev, U);
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_rows; j++)
        {
            double sq = sqrt(MATRIX_IDX_INTO(dev, i, j));
            MATRIX_IDX_INTO(tmp, i, j) = sq;
        }
    }

    // matrix* maha = matrix_multiply(dev, U);
    // return -0.5 * (rank * _LOG_2PI + log_det_cov + maha)) {
}

Data::Data(float *data, int num_records, int D, bool is_transposed) : data(data), num_records(num_records), D(D), is_transposed(is_transposed) {}

Data Data::T(bool is_view) // not using it now. Use in future
{
    return Data(data, num_records, D, is_view);
}
Data::Row Data::operator[](int index)
{
    if (is_transposed)
    {
        return Row(data + index * num_records);
    }
    else
    {
        float *row = init_array(D, 0);
        for (int i = 0; i < D; i++)
        {
            row[i] = data[i * num_records + index];
        }
        return Row(row);
    }
}

void EM(Data data, int n_components, float **mixture_means = NULL, float **mixture_covs = NULL, float **mixture_weights = NULL)
{
    float *means = mixture_means == NULL ? NULL : *mixture_means;
    // expectation step
    if (means == NULL)
    {
        means = init_array(n_components * data.D, 1.0 / n_components);
        mixture_means = &means;
    }

    float *covs = mixture_covs == NULL ? NULL : *mixture_covs;
    if (covs == NULL)
    {
        covs = init_array(n_components * data.D * data.D, 0);
        for (int i = 0; i < n_components; i++)
        {
            for (int j = 0; j < data.D; j++)
            {
                covs[i * data.D * data.D + j * data.D + j] = 1.0;
            }
        }
        mixture_covs = &covs;
    }

    float *weights = mixture_weights == NULL ? NULL : *mixture_weights;
    if (weights == NULL)
    {
        weights = init_array(n_components, 1.0 / n_components);
        mixture_weights = &weights;
    }

    float *responsibilities = init_array(data.num_records * n_components, 0);
    for (int j = 0; j < n_components; j++)
    {
        for (int i = 0; i < data.num_records; i++)
        {
            responsibilities[i * n_components + j] = weights[j] * pdf(&data.data[i * data.D], &means[j * data.D], &covs[j * data.D * data.D], data.D);
        }

        for (int i = 0; i < data.num_records; i++)
        {
            responsibilities[i * n_components + j] /= sum(&responsibilities[i * n_components], n_components);
        }
    }

    // maximization step
    float *sum_responsibilities = init_array(n_components, 0);
    for (int j = 0; j < n_components; j++)
    {
        for (int i = 0; i < data.num_records; i++)
        {
            sum_responsibilities[j] += responsibilities[i * n_components + j] / data.num_records;
        }
    }

    means = matrix_multiply(responsibilities, sum_responsibilities, data.num_records, n_components, n_components, 1);

    for (int i = 0; i < n_components; ++i)
    {
        float *temp = init_array(data.num_records * data.D, 0);
        memcpy(temp, data.data, data.num_records * data.D * sizeof(float));
        float *temp2 = init_array(data.num_records, 0);
        temp2 = calculate_self_dot_product(temp, data.num_records, data.D);
        // super unsure
        covs[i * data.D * data.D] = sum(matrix_multiply(responsibilities, temp2, data.num_records, n_components, n_components, 1), data.num_records) / sum_responsibilities[i];
    }
}

/**
 * Compute probability for vector x to belong to distribution. Generated, so pretty unsure about code.
 *
 *
 * ........
 */
float pdf(float *x, float *means, float *sigma, int dimension)
{
    float det;
    float *temp = (float *)malloc(dimension * sizeof(float));
    for (int i = 0; i < dimension; ++i)
    {
        det *= sigma[i * dimension + i];
        temp[i] = x[i] - means[i];
    }

    float exp = 0.0;
    float *inverse = invert_matrix(sigma, dimension);
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
float *invert_matrix(float *original, int dimension)
{
    float determinant = calculate_determinant(original, dimension);
    if (determinant == 0.0)
    {
        // Matrix is singular (non-invertible)
        throw std::invalid_argument("received non invertible matrix");
    }

    float d = 1. / determinant;

    float *inverse = init_array(dimension * dimension, 0);

    float *cofactor = calculate_cofactor(original, dimension);

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            float sign = pow(-1, i + j);
            inverse[i * dimension + j] = d * sign * cofactor[i * dimension + j];
        }
    }
    inverse = transpose(inverse, dimension, dimension);
    return inverse;
}

void print_matrix(float *matrix, int row_count, int col_count)
{
    for (int i = 0; i < row_count; i++)
    {
        for (int j = 0; j < col_count; j++)
        {
            printf("%f ", matrix[i * col_count + j]);
        }
        printf("\n");
    }
}

float *read_csv(char *filename, int row_count, int col_count, const char *delim)
{

    float *matrix = (float *)malloc(row_count * col_count * sizeof(float));

    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    char *line = NULL;
    size_t len = 0;
    int i = 0;
    while (getline(&line, &len, fp) != -1)
    {
        if (len == 0)
        {
            continue;
        }
        char *token = strtok(line, delim);
        int j = 0;
        while (token != NULL)
        {
            matrix[i * col_count + j] = atof(token);
            token = strtok(NULL, delim);
            j++;
        }
        i++;
    }
    fclose(fp);
    if (line)
    {
        free(line);
    }
    return matrix;
}

void store_csv(char *filename, float *matrix, int row_count, int col_count, const char *delim)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    for (int i = 0; i < row_count; i++)
    {
        int j = 0;
        for (j = 0; j < col_count - 1; j++)
        {
            fprintf(fp, "%f%s", matrix[i * col_count + j], delim);
        }
        fprintf(fp, "%f\n", matrix[i * col_count + j]);
    }
    fclose(fp);
}

// copilot generated
float calculate_determinant(float *matrix, int dimension)
{
    float det = 0;
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
            float *temp = init_array((dimension - 1) * (dimension - 1), 0);
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

struct GmmHyperparams
{
    float alpha;
    float beta;
    float W;
    float nu;
    float m;
};

float *init_array(int size, float value)
{
    float *array = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        array[i] = value;
    }
    return array;
}

float sum(float *array, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum;
}

float *matrix_multiply(float *matrix1, float *matrix2, int row_count1, int col_count1, int row_count2, int col_count2)
{
    float *result = init_array(row_count1 * col_count2, 0);
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

float *calculate_self_dot_product(float *matrix, int row_count, int col_count)
{
    float *result = init_array(row_count * col_count, 0);
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

float *transpose(float *matrix, int row_count, int col_count)
{
    float *result = init_array(row_count * col_count, 0);
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
float *calculate_cofactor(float *original, int dimension)
{
    float *cofactor = init_array(dimension * dimension, 0);
    float *temp = init_array((dimension - 1) * (dimension - 1), 0);
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

int main(int argc, char **argv)
{
    char *filename = argv[1];
    int n_components = atoi(argv[2]);
    int row_count = atoi(argv[3]);
    int D = atoi(argv[4]);
    char *means_fname = argv[5];
    char *covs_fname = argv[6];
    int max_iter = atoi(argv[7]);
    float *matrix = read_csv(filename, row_count, D);
    float *means;
    float *covs;
    float *weights;
    for (int i = 0; i < max_iter; i++)
    {
        EM(Data(matrix, row_count, D), n_components, &means, &covs, &weights);
    }
    store_csv(means_fname, means, n_components, D);
    store_csv(covs_fname, covs, n_components, D * D);

    return 0;
}