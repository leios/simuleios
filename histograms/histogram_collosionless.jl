using Test
using CUDA
using KernelAbstractions

if has_cuda_gpu()
    using CUDAKernels
end

# Function to use as a baseline for CPU metrics
function create_histogram(input)
    histogram_output = zeros(Int, maximum(input))
    for i = 1:length(input)
        histogram_output[input[i]] += 1
    end
    return histogram_output
end

# This is for a 1D histogram, for n-dim binning, we need to think a bit more
# The problem here is that the @synchronize calls are only synchronizing across
#     blocks, while we need a way to combine all the shmem histograms together 
#     into a big global blob. This can be done by:
@kernel function histogram_collisionless_kernel!(histogram_output, input)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    @uniform warpsize = Int(32)
    @uniform gs = @groupsize()[1]

    @uniform N = length(histogram_output)

    shared_histogram = @localmem Int (gs, warpsize)

    @inbounds shared_histogram[lid,:] .= 0
    @synchronize()

    @uniform max_element = 0
    for min_element = 1:gs:N

        shared_bin = input[tid] - min_element + 1

        max_element = min_element + gs
        if max_element > N+1
            max_element = N+1
        end

        @inbounds shared_histogram[shared_bin, ((lid-1)%warpsize)+1] += 1

        temp = 0
        for i = 1:warpsize
            @inbounds temp += shared_histogram[lid,i]
        end

        @inbounds histogram_output[lid] += temp
    end

end

function histogram_collisionless(input; numcores = 4, numthreads = 256)
    # I don't know how maximum is optimized on the GPU
    if isa(input, Array)
        histogram_output = zeros(maximum(input))
        event = histogram_collisionless!(histogram_output, input;
                                         numcores = numcores,
                                         numthreads = numthreads)
        wait(event)
        return histogram_output
    else
        histogram_output = CuArray(zeros(maximum(input)))
        event = histogram_collisionless!(histogram_output, input;
                                         numcores = numcores,
                                         numthreads = numthreads)
        wait(event)
        return histogram_output
    end
end

function histogram_collisionless!(histogram_output, input;
                                  numcores = 4, numthreads = 256)

    if isa(input, Array)
        kernel! = histogram_collisionless_kernel!(CPU(), numcores)
    else
        kernel! = histogram_collisionless_kernel!(CUDADevice(), numthreads)
    end

    kernel!(histogram_output, input, ndrange=size(input))
end

#=
@testset "histogram tests" begin

    a = [rand(1:128) for i = 1:1000]
    final_histogram = histogram(a)

    CPU_collisionless_histogram = histogram_collisionless(a)

    @test isapprox(CPU_collisionless_histogram, final_histogram)

    if has_cuda_gpu()
        CUDA.allowscalar(false)
        d_a = CuArray(a)
        GPU_collisionless_histogram = histogram_collisionless(d_a)
        @test isapprox(Array(GPU_collisionless_histogram), final_histogram)
    end
end
=#
