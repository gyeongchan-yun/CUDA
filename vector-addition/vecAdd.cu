#include <stdio.h>
#include <stdlib.h>

/* compile: nvcc vecAdd.cu -o vecAdd */

__global__ void add_vec(float *A, float *B, float *C) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  C[i] = A[i] + B[i];
}


void init(float *V, int vec_size) {
  for (int i = 0; i < vec_size; i++) {
    V[i] = i;
  }
}

void verify(float *A, float *B, float *C, int vec_size) {
  for (int i = 0; i < vec_size; i++) {
    if (A[i] + B[i] != C[i]) {
      printf("Verification failed! A[%d] = %d, B[%d] = %d, C[%d] = %d\n",
             i, A[i], i, B[i], i, C[i]);
      return;
    }
  }
  printf("Verification success!\n");
}

int main() {
  int vec_size = 16384;

  float *A = (float*)malloc(sizeof(float) * vec_size); 
  float *B = (float*)malloc(sizeof(float) * vec_size); 
  float *C = (float*)malloc(sizeof(float) * vec_size);

  init(A, vec_size);
  init(B, vec_size);

  // Memory objects of the device
  float *d_A, *d_B, *d_C;
  size_t mem_obj_size = sizeof(float) * vec_size;

  // allocate memory objects d_A, d_B, and d_C.
  cudaMalloc(&d_A, mem_obj_size);
  cudaMalloc(&d_B, mem_obj_size);
  cudaMalloc(&d_C, mem_obj_size);

  // Copy "A" to "d_A" and copy "B" to "d_B" (host to device).
  cudaMemcpy(d_A, A, mem_obj_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, mem_obj_size, cudaMemcpyHostToDevice);

  // Launch the kernel.
  /*
     __ __ __ 
    |__|__|__| 
    |__|__|__|
    
    dim_block = (3,2)
    dim_grid = # of threads in each block
  */
  dim3 dim_block(32, 1);
  dim3 dim_grid(vec_size / 32, 1);
  add_vec<<< dim_grid, dim_block >>> (d_A, d_B, d_C);

  // Copy "d_C" to "C" (device to host).
  cudaMemcpy(C, d_C, mem_obj_size, cudaMemcpyDeviceToHost);

  verify(A, B, C, vec_size);

  // Release d_A, d_B, and d_C.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(A);
  free(B);
  free(C);

  return 0;
}

