#include "matrix_operations.cuh"

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

__global__ void matrix_power(double *A, int n_rows, int n_cols)
{
    __shared__ double A_shared[TILE_DIM][TILE_DIM];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    A_shared[threadIdx.y][threadIdx.x] = A[i * n_cols + j];

    if (i < n_rows && j < n_cols)
    {
        A[i * n_cols + j] = A[i * n_cols + j] * A[i * n_cols + j];
    }
}

__global__ void elementwise_pow(double *A, int n_row, int n_col, int power)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_row && j < n_col)
    {
        A[i * n_col + j] = pow(A[i * n_col + j], power);
    }
}

__global__ void elementwise_exp(double *A, int n_row, int n_col)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_row && j < n_col)
    {
        A[i * n_col + j] = exp(A[i * n_col + j]);
    }
}

__global__ void transpose(double *odata, double *idata, int width, int height)
{
    __shared__ double block[TILE_DIM][TILE_DIM + 1];

    // read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
    unsigned int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y_index = blockIdx.y * TILE_DIM + threadIdx.y;
    if ((x_index < width) && (y_index < height))
    {
        unsigned int index_in = y_index * width + x_index;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (odata) in linear order
    x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    y_index = blockIdx.y * TILE_DIM + threadIdx.y;
    if ((x_index < height) && (y_index < width))
    {
        unsigned int index_out = y_index * height + x_index;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}
/**
 * Reduces matrix along rows and returns vector of row sums
 */
__global__ void reduce_matrix_rows(double *d_matrix, int n_rows, int n_cols, double *d_result)
{
    __shared__ double block[TILE_DIM][TILE_DIM + 1];

    // read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
    unsigned int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y_index = blockIdx.y * TILE_DIM + threadIdx.y;
    if ((x_index < n_cols) && (y_index < n_rows))
    {
        unsigned int index_in = y_index * n_cols + x_index;
        block[threadIdx.y][threadIdx.x] = d_matrix[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    double sum = 0;

    for (int j = 0; j < n_cols; j++)
    {
        sum += block[threadIdx.y][j];
    }

    if (threadIdx.x == 0)
    {
        d_result[y_index] += sum;
    }
}