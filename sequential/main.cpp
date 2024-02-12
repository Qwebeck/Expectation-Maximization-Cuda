#include "../linalg/eigen.h"
#include "EM.h"
#include "utils.h"

/*
TODO:
[ ] parallelize for CUDA
-   read the paper
-   understand linalg object
[ ] benchmark Sequential version

*/

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

    matrix *means = NULL;
    matrix **covs = NULL;
    vector *weights = NULL;

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