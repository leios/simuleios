#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <math.h>

// OpenCL kernel
const char *kernelSource =                          "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable       \n" \
"__kernel void vecAdd( __global double *a,           \n" \
"                      __global double *b,           \n" \
"                      __global double *c,           \n" \
"                      const unsigned int n){        \n" \
"                                                    \n" \
"    // Global Tread ID                              \n" \
"    int id = get_global_id(0);                      \n" \
"                                                    \n" \
"    // Remain in boundaries                         \n" \
"    if (id < n){                                    \n" \
"        c[id] = a[id] + b[id];                      \n" \
"    }                                               \n" \
"}                                                   \n";

int main(){
    unsigned int n = 1000;

    double *h_a, *h_b, *h_c;

    h_a = new double[n];
    h_b = new double[n];
    h_c = new double[n];

    for (size_t i = 0; i < n; ++i){
        h_a[i] = 1;
        h_b[i] = 1;
    }

    cl::Buffer d_a, d_b, d_c;

    cl_int err = CL_SUCCESS;
    try{
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(platforms.size() == 0){
            std::cout << "Platforms size is 0\n";
            return -1;
        }

        cl_context_properties properties[] = 
            { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };

        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::CommandQueue queue(context, devices[0], 0, &err);

        d_a = cl::Buffer(context, CL_MEM_READ_ONLY, n*sizeof(double));
        d_b = cl::Buffer(context, CL_MEM_READ_ONLY, n*sizeof(double));
        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, n*sizeof(double));

        queue.enqueueWriteBuffer(d_a, CL_TRUE, 0, n*sizeof(double), h_a);
        queue.enqueueWriteBuffer(d_b, CL_TRUE, 0, n*sizeof(double), h_b);

        cl::Program::Sources source(1,
            std::make_pair(kernelSource,strlen(kernelSource)));
        cl::Program program_ = cl::Program(context, source);
        program_.build(devices);

        cl::Kernel kernel(program_, "vecAdd", &err);

        kernel.setArg(0, d_a);
        kernel.setArg(1, d_b);
        kernel.setArg(2, d_c);
        kernel.setArg(3, n);

        cl::NDRange localSize(64);

        cl::NDRange globalSize((int)(ceil(n/(float)64)*64));

        cl::Event event;
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            globalSize,
            localSize,
            NULL,
            &event
        );

        event.wait();
        queue.enqueueReadBuffer(d_c, CL_TRUE, 0, n*sizeof(double), h_c);
    }
    catch(cl::Error err){
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")\n";
    }

    // Check to make sure everything works
    for (size_t i = 0; i < n; ++i){
        if (h_c[i] != h_a[i] + h_b[i]){
            std::cout << "Yo. You failed. What a loser! Ha\n";
            exit(1);
        }
    }

    std::cout << "You passed the test, congratulations!\n";

    delete(h_a);
    delete(h_b);
    delete(h_c);
}
