#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // OpenMP header

#define N 1000  // Matrix size (N x N)

// To initialize matrices with random values
void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (double)rand() / RAND_MAX;
        }
    }
}

// To perform parallel matrix multiplication using OpenMP
void matrix_multiply_omp(double *A, double *B, double *C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0.0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

// Loop ordering optimization (i-k-j order for better cache locality)
void matrix_multiply_omp_optimized(double *A, double *B, double *C, int size, int num_threads) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0.0;
        }
        
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main() {
    double *A, *B, *C;
    double start_time, end_time;
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
    int num_thread_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    // Allocate memory for matrices
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // Seed the random number generator
    srand(42);

    // Initialize matrices
    initialize_matrix(A, N);
    initialize_matrix(B, N);

    printf("Matrix size: %d x %d\n", N, N);
    
    // Standard implementation
    printf("\nStandard implementation:\n");
    printf("Threads\tTime(s)\tSpeedup\n");
    
    // Measure sequential time first (1 thread)
    start_time = omp_get_wtime();
    matrix_multiply_omp(A, B, C, N, 1);
    end_time = omp_get_wtime();
    double seq_time = end_time - start_time;
    printf("%d\t%.4f\t%.2f\n", 1, seq_time, 1.0);
    
    // Run with different thread counts
    for (int i = 1; i < num_thread_configs; i++) {
        int threads = thread_counts[i];
        start_time = omp_get_wtime();
        matrix_multiply_omp(A, B, C, N, threads);
        end_time = omp_get_wtime();
        double time_spent = end_time - start_time;
        printf("%d\t%.4f\t%.2f\n", threads, time_spent, seq_time/time_spent);
    }
    
    // Cache optimized implementation
    printf("\nCache-optimized implementation (i-k-j):\n");
    printf("Threads\tTime(s)\tSpeedup\n");
    
    // Measure sequential time for optimized version
    start_time = omp_get_wtime();
    matrix_multiply_omp_optimized(A, B, C, N, 1);
    end_time = omp_get_wtime();
    seq_time = end_time - start_time;
    printf("%d\t%.4f\t%.2f\n", 1, seq_time, 1.0);
    
    // Run optimized version with different thread counts
    for (int i = 1; i < num_thread_configs; i++) {
        int threads = thread_counts[i];
        start_time = omp_get_wtime();
        matrix_multiply_omp_optimized(A, B, C, N, threads);
        end_time = omp_get_wtime();
        double time_spent = end_time - start_time;
        printf("%d\t%.4f\t%.2f\n", threads, time_spent, seq_time/time_spent);
    }

    // Clean up and free allocated memory (key)
    free(A);
    free(B);
    free(C);
    
    return 0;
}