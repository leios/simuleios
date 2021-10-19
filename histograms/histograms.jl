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

# This is for a 1D histogram, for n-dim binning, we need to think a bit more
# The problem here is that the @synchronize calls are only synchronizing across
#     blocks, while we need a way to combine all the shmem histograms together 
#     into a big global blob. This can be done by:
#     1. making sure we use lid instead of tid for the histogram at the end
#        and use atomic adds
#     2. lid doesn't make sense on the CPU with only 4 cores.
#     3. fix error with jl_apply_type for GPU
#     4. Think about svectors
@kernel function histogram_tag_kernel!(histogram, values)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    shared_histogram = @localmem Int32 length(histogram)

    if tid == 1
        @print("LENGTH OF SHARED HIST: ", length(shared_histogram), "\n")
    end

    # Note: this is 4 for CPU
    @print("LID is: ", lid, '\n')

    # TODO: anything but this!
    if lid <= length(histogram)
        @inbounds shared_histogram[lid] = 0
    end

    @synchronize()

    #@print("values are: ", values[tid], "\n")

    if tid < length(values)
        @inbounds shared_histogram[values[tid]] += 1
    end 

    @synchronize()

    if tid <= length(histogram)
        #@print("histogram values: ", shared_histogram[tid], '\n')
        @inbounds histogram[tid] += shared_histogram[lid]
    end
end

function histogram_tag(values; numcores = 4, numthreads = 256)
    # I don't know how maximum is optimized on the GPU
    histogram = zeros(Int32, maximum(values))
    if isa(values, Array)
        event = histogram_tag!(histogram, values;
                               numcores = numcores, numthreads = numthreads)
    else
        event = histogram_tag!(CuArray(histogram), values;
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

    kernel!(histogram, values, ndrange=size(values))
end

#=
@testset "histogram tests" begin

    a = [Int32(rand(1:128)) for i = 1:1000]
    final_histogram = histogram(a)

    CPU_tag_histogram = histogram_tag(a)

    @test isapprox(CPU_tag_histogram, final_histogram)

    if has_cuda_gpu()
        CUDA.allowscalar(false)
        d_a = CuArray(a)
        GPU_tag_histogram = histogram_tag(d_a)
        @test isapprox(Array(GPU_tag_histogram), final_histogram)
    end
end
=#
