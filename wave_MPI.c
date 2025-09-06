#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define PI 3.14159265358979323846

// Grid parameters
#define DX 0.01      // Grid spacing
#define DY 0.01
#define DT 0.005     // Time step
#define C 1.0        // Wave speed
#define T_MAX 1.0    // Maximum simulation time
#define NUM_STEPS (int)(T_MAX/DT)  // Number of time steps

// Function to initialize local grid with initial conditions
void initialize_grid(double **u_curr, double **u_prev, 
                     int local_n_x, int local_n_y, 
                     int start_x, int start_y, 
                     int global_n_x, int global_n_y) {
    
    for (int i = 1; i <= local_n_x; i++) {
        double x = (start_x + i - 1) * DX;
        for (int j = 1; j <= local_n_y; j++) {
            double y = (start_y + j - 1) * DY;
            
            // Check if this is a boundary point
            if (start_x + i - 1 == 0 || start_x + i - 1 == global_n_x - 1 ||
                start_y + j - 1 == 0 || start_y + j - 1 == global_n_y - 1) {
                u_curr[i][j] = 0.0;
                u_prev[i][j] = 0.0;
            } else {
                // Initial displacement: sin(πx)sin(πy)
                u_curr[i][j] = sin(PI * x) * sin(PI * y);
                
                // Initial velocity = 0, we can compute u_prev using the scheme
                double laplacian = (sin(PI * (x + DX)) * sin(PI * y) 
                                   + sin(PI * (x - DX)) * sin(PI * y)
                                   + sin(PI * x) * sin(PI * (y + DY))
                                   + sin(PI * x) * sin(PI * (y - DY))
                                   - 4 * sin(PI * x) * sin(PI * y)) / (DX * DY);
                
                u_prev[i][j] = u_curr[i][j] + 0.5 * C * C * DT * DT * laplacian;
            }
        }
    }
}

// Function to update grid using finite difference scheme
void update_grid(double **u_next, double **u_curr, double **u_prev, 
                 int local_n_x, int local_n_y) {
    
    double factor = C * C * DT * DT / (DX * DY);
    
    for (int i = 1; i <= local_n_x; i++) {
        for (int j = 1; j <= local_n_y; j++) {
            u_next[i][j] = 2.0 * u_curr[i][j] - u_prev[i][j] 
                          + factor * (u_curr[i+1][j] + u_curr[i-1][j] 
                                     + u_curr[i][j+1] + u_curr[i][j-1] 
                                     - 4.0 * u_curr[i][j]);
        }
    }
}

// Function to exchange ghost cells with neighboring processes
void exchange_ghost_cells(double **u, int local_n_x, int local_n_y, 
                          int rank, int size, int dims[2], int coords[2], 
                          MPI_Comm cart_comm) {
    
    MPI_Status status;
    int neighbors[4];  // left, right, top, bottom
    
    // Find neighboring ranks in cartesian grid
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[1]);  // left, right
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]);  // top, bottom
    
    // Temporary buffers for sending/receiving
    double *send_left = (double*)malloc((local_n_y+2) * sizeof(double));
    double *send_right = (double*)malloc((local_n_y+2) * sizeof(double));
    double *recv_left = (double*)malloc((local_n_y+2) * sizeof(double));
    double *recv_right = (double*)malloc((local_n_y+2) * sizeof(double));
    
    double *send_top = (double*)malloc((local_n_x+2) * sizeof(double));
    double *send_bottom = (double*)malloc((local_n_x+2) * sizeof(double));
    double *recv_top = (double*)malloc((local_n_x+2) * sizeof(double));
    double *recv_bottom = (double*)malloc((local_n_x+2) * sizeof(double));
    
    // Copy data to send buffers
    for (int j = 0; j <= local_n_y+1; j++) {
        send_left[j] = u[1][j];
        send_right[j] = u[local_n_x][j];
    }
    
    for (int i = 0; i <= local_n_x+1; i++) {
        send_top[i] = u[i][1];
        send_bottom[i] = u[i][local_n_y];
    }
    
    // Exchange with left and right neighbors
    if (neighbors[0] != MPI_PROC_NULL) {
        MPI_Sendrecv(send_left, local_n_y+2, MPI_DOUBLE, neighbors[0], 0,
                    recv_left, local_n_y+2, MPI_DOUBLE, neighbors[0], 0,
                    cart_comm, &status);
        
        for (int j = 0; j <= local_n_y+1; j++) {
            u[0][j] = recv_left[j];
        }
    }
    
    if (neighbors[1] != MPI_PROC_NULL) {
        MPI_Sendrecv(send_right, local_n_y+2, MPI_DOUBLE, neighbors[1], 0,
                    recv_right, local_n_y+2, MPI_DOUBLE, neighbors[1], 0,
                    cart_comm, &status);
        
        for (int j = 0; j <= local_n_y+1; j++) {
            u[local_n_x+1][j] = recv_right[j];
        }
    }
    
    // Exchange with top and bottom neighbors
    if (neighbors[2] != MPI_PROC_NULL) {
        MPI_Sendrecv(send_top, local_n_x+2, MPI_DOUBLE, neighbors[2], 0,
                    recv_top, local_n_x+2, MPI_DOUBLE, neighbors[2], 0,
                    cart_comm, &status);
        
        for (int i = 0; i <= local_n_x+1; i++) {
            u[i][0] = recv_top[i];
        }
    }
    
    if (neighbors[3] != MPI_PROC_NULL) {
        MPI_Sendrecv(send_bottom, local_n_x+2, MPI_DOUBLE, neighbors[3], 0,
                    recv_bottom, local_n_x+2, MPI_DOUBLE, neighbors[3], 0,
                    cart_comm, &status);
        
        for (int i = 0; i <= local_n_x+1; i++) {
            u[i][local_n_y+1] = recv_bottom[i];
        }
    }
    
    // Free temporary buffers
    free(send_left);
    free(send_right);
    free(recv_left);
    free(recv_right);
    free(send_top);
    free(send_bottom);
    free(recv_top);
    free(recv_bottom);
}

// Function to save grid to file
void save_grid(double **local_grid, int local_n_x, int local_n_y,
               int start_x, int start_y, int global_n_x, int global_n_y,
               int rank, int size, int timestep, MPI_Comm cart_comm) {
    
    // Only rank 0 will write to file
    if (rank == 0) {
        double **global_grid = (double**)malloc(global_n_x * sizeof(double*));
        for (int i = 0; i < global_n_x; i++) {
            global_grid[i] = (double*)malloc(global_n_y * sizeof(double));
        }
        
        // Copy local data to global grid
        for (int i = 1; i <= local_n_x; i++) {
            for (int j = 1; j <= local_n_y; j++) {
                global_grid[start_x + i - 1][start_y + j - 1] = local_grid[i][j];
            }
        }
        
        // Receive data from other processes
        for (int p = 1; p < size; p++) {
            MPI_Status status;
            int recv_coords[2], recv_start[2], recv_local_n[2];
            
            // Receive process coordinates and local grid dimensions
            MPI_Recv(recv_coords, 2, MPI_INT, p, 0, cart_comm, &status);
            MPI_Recv(recv_start, 2, MPI_INT, p, 1, cart_comm, &status);
            MPI_Recv(recv_local_n, 2, MPI_INT, p, 2, cart_comm, &status);
            
            // Receive grid data
            double *recv_buffer = (double*)malloc(recv_local_n[0] * recv_local_n[1] * sizeof(double));
            MPI_Recv(recv_buffer, recv_local_n[0] * recv_local_n[1], MPI_DOUBLE, p, 3, cart_comm, &status);
            
            // Copy received data to global grid
            for (int i = 0; i < recv_local_n[0]; i++) {
                for (int j = 0; j < recv_local_n[1]; j++) {
                    global_grid[recv_start[0] + i][recv_start[1] + j] = recv_buffer[i * recv_local_n[1] + j];
                }
            }
            
            free(recv_buffer);
        }
        
        // Write global grid to file
        char filename[100];
        sprintf(filename, "wave_mpi_step_%d.dat", timestep);
        
        FILE *file = fopen(filename, "w");
        if (file == NULL) {
            printf("Error opening file\n");
            return;
        }
        
        for (int i = 0; i < global_n_x; i++) {
            for (int j = 0; j < global_n_y; j++) {
                fprintf(file, "%f ", global_grid[i][j]);
            }
            fprintf(file, "\n");
        }
        
        fclose(file);
        
        // Free global grid
        for (int i = 0; i < global_n_x; i++) {
            free(global_grid[i]);
        }
        free(global_grid);
    } 
    else {
        // Send local data to rank 0
        int coords[2], start[2], local_n[2];
        coords[0] = start_x;
        coords[1] = start_y;
        start[0] = start_x;
        start[1] = start_y;
        local_n[0] = local_n_x;
        local_n[1] = local_n_y;
        
        MPI_Send(coords, 2, MPI_INT, 0, 0, cart_comm);
        MPI_Send(start, 2, MPI_INT, 0, 1, cart_comm);
        MPI_Send(local_n, 2, MPI_INT, 0, 2, cart_comm);
        
        // Pack local grid data (excluding ghost cells)
        double *send_buffer = (double*)malloc(local_n_x * local_n_y * sizeof(double));
        for (int i = 1; i <= local_n_x; i++) {
            for (int j = 1; j <= local_n_y; j++) {
                send_buffer[(i-1) * local_n_y + (j-1)] = local_grid[i][j];
            }
        }
        
        MPI_Send(send_buffer, local_n_x * local_n_y, MPI_DOUBLE, 0, 3, cart_comm);
        
        free(send_buffer);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    int global_n_x, global_n_y;
    int local_n_x, local_n_y;
    int start_x, start_y;
    double **u_curr, **u_prev, **u_next, **temp;
    double start_time, end_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get problem size from command line if provided
    if (argc > 1) {
        global_n_x = global_n_y = atoi(argv[1]);
    } else {
        global_n_x = global_n_y = 100;  // Default size
    }
    
    // Create 2D Cartesian communicator
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};  // Non-periodic boundaries
    MPI_Dims_create(size, 2, dims);
    
    if (rank == 0) {
        printf("Using %d processes in a %d x %d grid\n", size, dims[0], dims[1]);
        printf("Global grid size: %d x %d\n", global_n_x, global_n_y);
    }
    
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Calculate local grid dimensions
    local_n_x = global_n_x / dims[0];
    local_n_y = global_n_y / dims[1];
    
    // Adjust for non-divisible grid sizes
    if (coords[0] < global_n_x % dims[0]) local_n_x++;
    if (coords[1] < global_n_y % dims[1]) local_n_y++;
    
    // Calculate starting indices in global grid
    start_x = coords[0] * (global_n_x / dims[0]);
    start_y = coords[1] * (global_n_y / dims[1]);
    
    if (coords[0] < global_n_x % dims[0]) {
        start_x += coords[0];
    } else {
        start_x += global_n_x % dims[0];
    }
    
    if (coords[1] < global_n_y % dims[1]) {
        start_y += coords[1];
    } else {
        start_y += global_n_y % dims[1];
    }
    
    if (rank == 0) {
        printf("Starting simulation with %d time steps\n", NUM_STEPS);
    }
    
    // Allocate memory for local grids (including ghost cells)
    u_curr = (double**)malloc((local_n_x+2) * sizeof(double*));
    u_prev = (double**)malloc((local_n_x+2) * sizeof(double*));
    u_next = (double**)malloc((local_n_x+2) * sizeof(double*));
    
    for (int i = 0; i <= local_n_x+1; i++) {
        u_curr[i] = (double*)malloc((local_n_y+2) * sizeof(double));
        u_prev[i] = (double*)malloc((local_n_y+2) * sizeof(double));
        u_next[i] = (double*)malloc((local_n_y+2) * sizeof(double));
    }
    
    // Initialize local grids
    initialize_grid(u_curr, u_prev, local_n_x, local_n_y, start_x, start_y, global_n_x, global_n_y);
    
    // Synchronize all processes before starting the timer
    MPI_Barrier(cart_comm);
    start_time = MPI_Wtime();
    
    // Main simulation loop
    for (int t = 1; t <= NUM_STEPS; t++) {
        // Exchange ghost cells
        exchange_ghost_cells(u_curr, local_n_x, local_n_y, rank, size, dims, coords, cart_comm);
        
        // Update grid
        update_grid(u_next, u_curr, u_prev, local_n_x, local_n_y);
        
        // Rotate pointers for next time step
        temp = u_prev;
        u_prev = u_curr;
        u_curr = u_next;
        u_next = temp;
        
        // Save grid state at regular intervals
        if (t % 50 == 0) {
            save_grid(u_curr, local_n_x, local_n_y, start_x, start_y, 
                     global_n_x, global_n_y, rank, size, t, cart_comm);
        }
    }
    
    // Synchronize before stopping the timer
    MPI_Barrier(cart_comm);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("MPI wave equation simulation (N=%d, steps=%d, procs=%d) took %f seconds\n", 
               global_n_x, NUM_STEPS, size, end_time - start_time);
    }
    
    // Save final state
    save_grid(u_curr, local_n_x, local_n_y, start_x, start_y, 
             global_n_x, global_n_y, rank, size, NUM_STEPS, cart_comm);
    
    // Clean up
    for (int i = 0; i <= local_n_x+1; i++) {
        free(u_curr[i]);
        free(u_prev[i]);
        free(u_next[i]);
    }
    free(u_curr);
    free(u_prev);
    free(u_next);
    
    MPI_Finalize();
    return 0;
}