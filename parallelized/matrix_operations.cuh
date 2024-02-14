#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#define TILE_DIM 32
__global__ void gemm_tiled(double *A, double *B, double *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols);

__global__ void matrix_power(double *A, int n_rows, int n_cols);
__global__ void elementwise_pow(double *A, int n_row, int n_col, int power);
__global__ void transpose(double *odata, double *idata, int width, int height);
__global__ void reduce_matrix_rows(double *d_matrix, int n_rows, int n_cols, double *d_result);
__global__ void elementwise_exp(double *A, int n_row, int n_col);

#endif