#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h>
#include "../linalg/eigen.h"

matrix **init_array(int size, matrix *value);
float *read_csv(char *filename, int row_count, int col_count, const char *delim = ",");
void store_csv(char *filename, float *matrix, int row_count, int col_count, const char *delim = ",");

float vector_sum(struct vector *v);
float min_value(float *array, int size);
float max_value(float *array, int size);
float mean_value(float *array, int size);
float std_value(float *array, int size);
char *describe(float *array, int size);
void print_array(float *array, int size);
void append_to_file(char *filename, char *text);

#endif