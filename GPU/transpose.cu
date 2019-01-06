#include<iostream>

__global__ void copy(double* a_in, double *a_out){
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    a_out[gid] = a_in[gid];
}

__global__ void simple_transpose(double* a_in, double *a_out){
    int gid_in = blockIdx.x*blockDim.x + threadIdx.x;
    int gid_out = threadIdx.x*blockDim.x + blockIdx.x;

    a_out[gid_in] = a_in[gid_out];
}

void transpose(double* a_in, double*a_out, int xDim, int yDim){
    for (int i = 0; i < xDim; ++i){
        for (int j = 0; j < yDim; ++j){
            int index_in = j + i*yDim;
            int index_out = i + j*xDim;
            a_out[index_in] = a_in[index_out];
        }
    }
}

void print_array(double *a, int xDim, int yDim){
    for (int i = 0; i < xDim; ++i){
        for (int j = 0; j < yDim; ++j){
            int index = j + i*yDim;
            std::cout << a[index];
            if (j != yDim - 1){
                std::cout << '\t';
            }
        }
        std::cout << '\n';
    }
}

int main(){

    double *a_in, *a_out;
    double *da_in, *da_out;
    unsigned int xDim = 8;
    unsigned int yDim = 8;
    unsigned int gSize = xDim*yDim;

    dim3 grid = {yDim, 1, 1};
    dim3 threads = {xDim, 1, 1};

    a_in = (double *)malloc(sizeof(double)*gSize);
    a_out = (double *)malloc(sizeof(double)*gSize);

    cudaMalloc((void**)&da_in, sizeof(double)*gSize);
    cudaMalloc((void**)&da_out, sizeof(double)*gSize);

    for (int i = 0; i < gSize; ++i){
        a_in[i] = i;
    }

    cudaMemcpy(da_in, a_in, sizeof(double)*gSize, cudaMemcpyHostToDevice);

    print_array(a_in, xDim, yDim);
    std::cout << '\n';

    //transpose(a_in, a_out, xDim, yDim);
    simple_transpose<<<grid, threads>>>(da_in, da_out);

    cudaMemcpy(a_out, da_out, sizeof(double)*gSize, cudaMemcpyDeviceToHost);
    print_array(a_out, xDim, yDim);
}
