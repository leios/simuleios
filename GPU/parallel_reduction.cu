#include <iostream>

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, int tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid+32];
    if (blockSize >= 32) sdata[tid] += sdata[tid+16];
    if (blockSize >= 16) sdata[tid] += sdata[tid+8];
    if (blockSize >= 8) sdata[tid] += sdata[tid+4];
    if (blockSize >= 4) sdata[tid] += sdata[tid+2];
    if (blockSize >= 2) sdata[tid] += sdata[tid+1];
}

__device__ void warpReduce(volatile int* sdata, int tid){
    sdata[tid] += sdata[tid+32];
    sdata[tid] += sdata[tid+16];
    sdata[tid] += sdata[tid+8];
    sdata[tid] += sdata[tid+4];
    sdata[tid] += sdata[tid+2];
    sdata[tid] += sdata[tid+1];
}

__global__ void reduce0(int *idata, int *odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2){
        if (tid % (2*s) == 0){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce1(int *idata, int *odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2){
        int index = 2 * s * tid;
        if (index < blockDim.x){
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }

    if (tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce2(int *idata, int *odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = idata[i];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce3(int *idata, int *odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = idata[i] +idata[i+blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s>>=1){ 
        if (tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce4(int *idata, int *odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = idata[i] +idata[i+blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 32; s>>=1){ 
        if (tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (tid < 32){
        warpReduce(sdata, tid);
    }

    if (tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__global__ void reduce5(int *idata, int *odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = idata[i] +idata[i+blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 32; s>>=1){ 
        if (tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }

    if (blockSize >= 512){
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256){
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128){
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32){
        warpReduce<blockSize>(sdata, tid);
    }

    if (tid == 0){
        odata[blockIdx.x] = sdata[0];
    }
}

double sum(int *a, int n){
    int output = 0;
    for (int i = 0; i < n; ++i){
        output += a[i];
    }

    return output;
}

int main(){

    int *in_a, *out_a;
    int *din_a, *dout_a;
    unsigned int n = 1024;
    in_a = (int*)malloc(sizeof(int)*n);
    out_a = (int*)malloc(sizeof(int)*n);

    cudaMalloc(&din_a, sizeof(int)*n);
    cudaMalloc(&dout_a, sizeof(int)*n);

    for (int i = 0; i < n; ++i){
        in_a[i] = 1;
    }

    cudaMemcpy(din_a, in_a, sizeof(int)*n, cudaMemcpyHostToDevice);

    int output = sum(in_a, n);
    std::cout << output << '\n';

    dim3 threads = {n, 1, 1};
    dim3 grid = {1, 1, 1};
    //reduce4<<<grid, threads, sizeof(int)*n>>>(din_a, dout_a);
    reduce5<64><<<grid, threads, sizeof(int)*n>>>(din_a, dout_a);

    cudaMemcpy(out_a, dout_a, sizeof(int)*n, cudaMemcpyDeviceToHost);

    std::cout << out_a[0] << '\n';

    free(in_a); free(out_a);
    cudaFree(din_a); cudaFree(dout_a);

}
