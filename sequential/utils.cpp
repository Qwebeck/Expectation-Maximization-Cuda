#include "utils.h"
#include <string.h>
#include "../linalg/vector.h"
#include <errno.h>
#include <math.h>
#include <fstream>
#include <iostream>

void append_to_file(char *filename, char *text)
{
    // Open the file in append mode. This will create the file if it does not exist.
    std::ofstream file;
    file.open(filename, std::ios::app);

    if (file.is_open())
    {
        // Write the text to the file
        file << text << std::endl;

        // Close the file
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

double *read_csv(char *filename, int row_count, int col_count, const char *delim)
{
    errno = 0;
    double *matrix = (double *)malloc(row_count * col_count * sizeof(double));
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    char *line = NULL;
    size_t len = 0;
    int i = 0;
    while (getline(&line, &len, fp) != -1)
    {
        if (len == 0)
        {
            continue;
        }
        char *token = strtok(line, delim);
        int j = 0;
        while (token != NULL)
        {
            matrix[i * col_count + j] = atof(token);
            token = strtok(NULL, delim);
            j++;
        }
        i++;
    }
    fclose(fp);
    if (line)
    {
        free(line);
    }
    return matrix;
}

void store_csv(char *filename, double *matrix, int row_count, int col_count, const char *delim)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }
    for (int i = 0; i < row_count; i++)
    {
        int j = 0;
        for (j = 0; j < col_count - 1; j++)
        {
            fprintf(fp, "%f%s", matrix[i * col_count + j], delim);
        }
        fprintf(fp, "%f\n", matrix[i * col_count + j]);
    }
    fclose(fp);
}

// template <typename T>
matrix **init_array(int size, matrix *value)
{
    matrix **result = (matrix **)malloc(size * sizeof(matrix *));
    for (int i = 0; i < size; i++)
    {
        result[i] = matrix_copy(value);
    }
    return result;
}

double vector_sum(struct vector *v)
{
    double sum = 0;
    for (int i = 0; i < v->length; i++)
    {
        sum += VECTOR_IDX_INTO(v, i);
    }
    return sum;
}

double min_value(double *array, int size)
{
    double min = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] < min)
        {
            min = array[i];
        }
    }
    return min;
}

double max_value(double *array, int size)
{
    double max = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max)
        {
            max = array[i];
        }
    }
    return max;
}

double mean_value(double *array, int size)
{
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum / size;
}

double std_value(double *array, int size)
{
    double m = mean_value(array, size);
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += pow(array[i] - m, 2);
    }
    return sqrt(sum / size);
}
char *describe(double *array, int size)
{
    double min = min_value(array, size);
    double max = max_value(array, size);
    double mean = mean_value(array, size);
    double std = std_value(array, size);
    char *result = (char *)malloc(100 * sizeof(char));
    sprintf(result, "min: %f, max: %f, mean: %f, std: %f", min, max, mean, std);
    return result;
}

void print_array(double *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n");
}