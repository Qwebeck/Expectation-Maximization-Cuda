#pragma once
#include "vector.h"
#include "matrix.h"


void init_random();
struct vector* vector_random_uniform(int length, float low, float high);
struct matrix* matrix_random_uniform(int n_row, int n_col, float low, float high);

struct vector* vector_random_gaussian(int length, float mu, float sigma);
