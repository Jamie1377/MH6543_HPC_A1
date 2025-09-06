#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

// Grid parameters
#define N 100        // Grid size (N x N)
#define DX 0.01      // Grid spacing
#define DY 0.01
#define DT 0.005     // Time step
#define C 1.0        // Wave speed
#define T_MAX 1.0    // Maximum simulation time
#define NUM_STEPS (int)(T_MAX/DT)  // Number of time steps

// Function to initialize grid with initial conditions
void initialize_grid(double **u_curr, double **u_prev, int n) {
    for (int i = 0; i < n; i++) {
        double x = i * DX;
        for (int j = 0; j < n; j++) {
            double y = j * DY;
            
            // Initial displacement: sin(πx)sin(πy)
            u_curr[i][j] = sin(PI * x) * sin(PI * y);
            
            // Initial velocity = 0, we can compute u_prev using the scheme:
            // u_prev = u_curr - dt * velocity + 0.5 * dt^2 * acceleration
            // Since velocity = 0, we just need to account for acceleration:
            double laplacian = (sin(PI * (x + DX)) * sin(PI * y) 
                               + sin(PI * (x - DX)) * sin(PI * y)
                               + sin(PI * x) * sin(PI * (y + DY))
                               + sin(PI * x) * sin(PI * (y - DY))
                               - 4 * sin(PI * x) * sin(PI * y)) / (DX * DY);
            
            u_prev[i][j] = u_curr[i][j] + 0.5 * C * C * DT * DT * laplacian;
        }
    }
    
    // Set boundary conditions to 0 (Dirichlet)
    for (int i = 0; i < n; i++) {
        u_curr[i][0] = u_curr[i][n-1] = 0.0;
        u_curr[0][i] = u_curr[n-1][i] = 0.0;
        u_prev[i][0] = u_prev[i][n-1] = 0.0;
        u_prev[0][i] = u_prev[n-1][i] = 0.0;
    }
}

// Function to update grid using finite difference scheme
void update_grid(double **u_next, double **u_curr, double **u_prev, int n) {
    double factor = C * C * DT * DT / (DX * DY);
    
    for (int i = 1; i < n-1; i++) {
        for (int j = 1; j < n-1; j++) {
            // Use the finite difference scheme:
            // u[i,j,n+1] = 2*u[i,j,n] - u[i,j,n-1] + c^2*dt^2/(dx*dy) * (u[i+1,j,n] + u[i-1,j,n] + u[i,j+1,n] + u[i,j-1,n] - 4*u[i,j,n])
            u_next[i][j] = 2.0 * u_curr[i][j] - u_prev[i][j] 
                          + factor * (u_curr[i+1][j] + u_curr[i-1][j] 
                                     + u_curr[i][j+1] + u_curr[i][j-1] 
                                     - 4.0 * u_curr[i][j]);
        }
    }
    
    // Apply boundary conditions
    for (int i = 0; i < n; i++) {
        u_next[i][0] = u_next[i][n-1] = 0.0;
        u_next[0][i] = u_next[n-1][i] = 0.0;
    }
}

// Function to save grid to file
void save_grid(double **u, int n, int timestep) {
    char filename[100];
    sprintf(filename, "wave_seq_step_%d.dat", timestep);
    
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file\n");
        return;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%f ", u[i][j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}

int main() {
    int n = N + 2;  // Include boundary points
    double **u_curr, **u_prev, **u_next, **temp;
    struct timespec start, end;
    double time_spent;
    
    // Allocate memory for grids
    u_curr = (double**)malloc(n * sizeof(double*));
    u_prev = (double**)malloc(n * sizeof(double*));
    u_next = (double**)malloc(n * sizeof(double*));
    
    for (int i = 0; i < n; i++) {
        u_curr[i] = (double*)malloc(n * sizeof(double));
        u_prev[i] = (double*)malloc(n * sizeof(double));
        u_next[i] = (double*)malloc(n * sizeof(double));
    }
    
    // Initialize grids
    initialize_grid(u_curr, u_prev, n);
    
    // Save initial state
    save_grid(u_curr, n, 0);
    
    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Main simulation loop
    for (int t = 1; t <= NUM_STEPS; t++) {
        // Update grid
        update_grid(u_next, u_curr, u_prev, n);
        
        // Rotate pointers for next time step
        temp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = temp;
        
        // Save grid state at regular intervals (e.g., every 50 steps)
        if (t % 50 == 0) {
            save_grid(u_curr, n, t);
        }
    }
    
    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("Sequential wave equation simulation (N=%d, steps=%d) took %f seconds\n", 
           N, NUM_STEPS, time_spent);
    
    // Save final state
    save_grid(u_curr, n, NUM_STEPS);
    
    // Clean up
    for (int i = 0; i < n; i++) {
        free(u_curr[i]);
        free(u_prev[i]);
        free(u_next[i]);
    }
    free(u_curr);
    free(u_prev);
    free(u_next);
    
    return 0;
}