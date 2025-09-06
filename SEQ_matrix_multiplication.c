#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000  // Matrix size (N x N) and N = 1000

// To initialize matrices with random values and make sure its dense not sparse
void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (double)rand() / RAND_MAX;
        }
    }
}

// To perform sequential matrix multiplication (Nested for loops) 
// C is the result matrix that stores the result of A*B
void matrix_multiply_seq(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0.0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main() {
    double *A, *B, *C;
    struct timespec start, end;
    double time_spent;

    // Allocate memory for matrices A, B, and C
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Seed the random number generator
    srand(42);

    // Initialize matrices A and B with size 1000 x 1000 
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    // Measure execution time
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Perform matrix multiplication
    matrix_multiply_seq(A, B, C, N);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("Sequential matrix multiplication (N=%d) took %f seconds\n", N, time_spent);

    // Clean up and free allocated memory (key)
    free(A);
    free(B);
    free(C);
    
    return 0;
}