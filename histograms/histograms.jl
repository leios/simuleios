using Test
using CUDA
using CUDAKernels
using KernelAbstractions

# Function to use as a baseline for CPU metrics
function histogram(values)
    histogram = zeros(Int, maximum(values))
    for i = 1:length(values)
        histogram[values[i]] += 1
    end
    return histogram
end

@kernel function histogram_tag_kernel!(histogram, values)
end

function histogram_tag(values; numcores = 4, numthreads = 256)
    # I don't know how maximum is optimized on the GPU
    histogram = zeros(Int, maximum(values))
    if has_cuda_gpu()
        event = histogram_tag!(CuArray(histogram), values;
                               numcores = numcores, numthreads = numthreads)
    else
        event = histogram_tag!(histogram, values;
                               numcores = numcores, numthreads = numthreads)
    end
    wait(event)
    return histogram
end

function histogram_tag!(histogram, values; numcores = 4, numthreads = 256)

    if isa(values, Array)
        kernel! = histogram_tag_kernel!(CPU(), numcores)
    else
        kernel! = histogram_tag_kernel!(CUDADevice(), numthreads)
    end

    kernel!(histogram, values, ndrange=size(histogram))
end

@testset "histogram tests" begin

    a = [rand(1:10) for i = 1:1000]
    final_histogram = histogram(a)

    CPU_tag_histogram = histogram_tag(a)

    @test isapprox(CPU_tag_histogram, final_histogram)

    if has_cuda_gpu()
        d_a = CuArray(a)
        GPU_tag_histogram = histogram_tag(d_a)
        @test isapprox(Array(GPU_tag_histogram), final_histogram)
    end
end
