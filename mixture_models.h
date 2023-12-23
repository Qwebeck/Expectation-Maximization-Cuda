#ifndef MIXTURE_MODELS_H
#define MIXTURE_MODELS_H

#include <stdio.h>
#include <stdlib.h>
class Data
{
public:
    float *data;
    int num_records;
    int D;
    bool is_transposed;

    Data(float *data, int num_records, int D, bool is_transposed = false);

    Data T(bool is_view);

    class Row
    {
    private:
        float *row;

    public:
        Row(float *row) : row(row) {}

        float &operator[](int index)
        {
            return row[index];
        }
    };
    Row operator[](int index);
};
// Forward declaration of functions used in mixture_models.cu
float *read_csv(char *filename, int row_count, int col_count, const char *delim = ",");
void store_csv(char *filename, float *matrix, int row_count, int col_count, const char *delim = ",");
float *init_array(int size, float value);
float *matrix_multiply(float *matrix1, float *matrix2, int row_count1, int col_count1, int row_count2, int col_count2);
float *calculate_self_dot_product(float *matrix, int row_count, int col_count);
float *invert_matrix(float *original, int dimension);
float sum(float *array, int size);
float pdf(float *x, float *means, float *sigma, int dimension);
void EM(Data data, int n_components, float **means, float **covs, float **weights);
void store_csv(const char *filename, float *data, int row_count, int D);
float calculate_determinant(float *matrix, int dimension);
float *calculate_cofactor(float *original, int dimension);

// Forward declaration of the Data class

#endif // MIXTURE_MODELS_H