#include <cufft.h>
#include <fstream>

int main(){
    // Initializing variables

    int n = 1024;
    cufftHandle plan1d;
    double2 *h_a, *d_a, *h_b;

    std::ofstream time_out("time_out.dat"), freq_out("freq_out.dat");

    // Allocations / definition
    h_a = (double2 *)malloc(sizeof(double2)*n);
    h_b = (double2 *)malloc(sizeof(double2)*n);
    for (int i = 0; i < n; ++i){
        h_a[i].x = sin(20*2*M_PI*i/n);
        h_a[i].y = 0;
    }

    cudaMalloc(&d_a, sizeof(double2)*n);
    cudaMemcpy(d_a, h_a, sizeof(double2)*n, cudaMemcpyHostToDevice);
    cufftPlan1d(&plan1d, n, CUFFT_Z2Z, 1);

    // FFT
    cufftExecZ2Z(plan1d, d_a, d_a, CUFFT_FORWARD);

    // Copying back
    cudaMemcpy(h_b, d_a, sizeof(double2)*n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i){
        time_out << h_a[i].x << '\n';
        freq_out << sqrt(h_b[i].x*h_b[i].x + h_b[i].y*h_b[i].y) << '\n';
    }

    time_out.close();
    freq_out.close();

}
