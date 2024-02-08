#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include "linalg/eigen.h"

matrix **init_array(int size, matrix *value);
double *read_csv(char *filename, int row_count, int col_count, const char *delim = ",");
void store_csv(char *filename, double *matrix, int row_count, int col_count, const char *delim = ",");

double vector_sum(struct vector *v);
double min_value(double *array, int size);
double max_value(double *array, int size);
double mean_value(double *array, int size);
double std_value(double *array, int size);
char *describe(double *array, int size);

#endif