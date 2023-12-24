#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <stdlib.h>

double *read_csv(char *filename, int row_count, int col_count, const char *delim = ",");
void store_csv(char *filename, float *matrix, int row_count, int col_count, const char *delim = ",");

#endif