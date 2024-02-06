#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include "linalg/eigen.h"

// template <typename T>
// T *init_array(int size, T value);

matrix **init_array(int size, matrix *value);
double *read_csv(char *filename, int row_count, int col_count, const char *delim = ",");
void store_csv(char *filename, double *matrix, int row_count, int col_count, const char *delim = ",");

double vector_sum(struct vector *v);

#endif