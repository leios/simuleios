using Test
using CUDA
using CUDAKernels
using KernelAbstractions

@kernel function copy_shared_kernel!(a,b)
    tid = @index(Global, Linear)

    #N = @uniform prod(@groupsize())

    #shared_tile = @localmem Int32 (N,)
    shared_tile = @localmem Int32 (10,)

    if tid <= length(a)
        @inbounds shared_tile[tid] = a[tid]
    end

    @synchronize()

    if tid <= length(a)
        @inbounds b[tid] = a[tid]
    end
end

function copy_shared!(a, b; numcores = 4, numthreads = 256)

    if isa(a, Array)
        kernel! = copy_shared_kernel!(CPU(), 4)
    else
        kernel! = copy_shared_kernel!(CUDADevice(), 256)
    end

    kernel!(a, b, ndrange=size(a))
end

@testset "CPU / GPU tests" begin
    a = Int64.(rand(1:128,10, 10))
    b = zeros(Int32, 10, 10)

    event = copy_shared!(a, b)
    wait(event)

    if has_cuda_gpu()
        d_a = CuArray(a)
        d_b = similar(d_a)

        event = copy_shared!(d_a, d_b)
        wait(event)

    end
end
