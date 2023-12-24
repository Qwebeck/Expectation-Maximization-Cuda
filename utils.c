#include "utils.h"

double *read_csv(char *filename, int row_count, int col_count, const char *delim)
{
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
