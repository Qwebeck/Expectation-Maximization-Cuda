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
    std::ofstream file(filename, std::ios::app);

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

float *read_csv(char *filename, int row_count, int col_count, const char *delim)
{
    errno = 0;
    float *matrix = (float *)malloc(row_count * col_count * sizeof(float));
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

void store_csv(char *filename, float *matrix, int row_count, int col_count, const char *delim)
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

float vector_sum(struct vector *v)
{
    float sum = 0;
    for (int i = 0; i < v->length; i++)
    {
        sum += VECTOR_IDX_INTO(v, i);
    }
    return sum;
}

float min_value(float *array, int size)
{
    float min = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] < min)
        {
            min = array[i];
        }
    }
    return min;
}

float max_value(float *array, int size)
{
    float max = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max)
        {
            max = array[i];
        }
    }
    return max;
}

float mean_value(float *array, int size)
{
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum / size;
}

float std_value(float *array, int size)
{
    float m = mean_value(array, size);
    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += pow(array[i] - m, 2);
    }
    return sqrt(sum / size);
}
char *describe(float *array, int size)
{
    float min = min_value(array, size);
    float max = max_value(array, size);
    float mean = mean_value(array, size);
    float std = std_value(array, size);
    char *result = (char *)malloc(100 * sizeof(char));
    sprintf(result, "min: %f, max: %f, mean: %f, std: %f", min, max, mean, std);
    return result;
}

void print_array(float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n");
}