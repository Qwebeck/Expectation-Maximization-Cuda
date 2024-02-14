#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#define TILE_DIM 32
__global__ void gemm_tiled(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);

__global__ void matrix_power(float *A, int n_rows, int n_cols);
__global__ void elementwise_pow(float *A, int n_row, int n_col, int power);
__global__ void transpose(float *odata, float *idata, int width, int height);
__global__ void reduce_matrix_rows(float *d_matrix, int n_rows, int n_cols, float *d_result);
__global__ void elementwise_exp(float *A, int n_row, int n_col);

#endif