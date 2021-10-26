using Test
using CUDA
using CUDAKernels
using KernelAbstractions

# Function to use as a baseline for CPU metrics
function histogram(input)
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
#     1. making sure we use lid instead of tid for the histogram at the end
#        and use atomic adds
#     2. lid doesn't make sense on the CPU with only 4 cores.
#     3. fix error with jl_apply_type for GPU
#     4. Think about svectors
#     5. dynamic shmem allocation is not allowed, so use threadsize or a const
#     6. We need to find a way to do a coalesced write to global memory
#         - I think this will be easier on the GPU when more threads are avail
@kernel function histogram_tag_kernel!(histogram_output, input)
    tid = @index(Global, Linear)
    #lid = @index(Local, Linear)

    #shared_histogram = @localmem Int32 length(histogram_output)
    shared_histogram = @localmem Int64 128

    # Note: this is 4 for CPU
    #@print("LID is: ", lid, '\n')
    @print("TID is: ", tid, '\n')

    # TODO: anything but this!
    if tid <= 128
        @inbounds shared_histogram[tid] = 0
    end

    @synchronize()

    #@print("values are: ", input[tid], "\n")

    if tid < length(histogram_output)
        @print("CHECK\n")
    end 

    if tid <= length(input)
        @inbounds temp = shared_histogram[input[tid]]
        @inbounds shared_histogram[input[tid]] = temp + 1
        @print("input is: ", input[tid], '\n')
        @print("hist is: ", shared_histogram[input[tid]], "\n\n")
    end 

    @synchronize()

    @print(length(histogram_output), '\n')

    if tid <= length(histogram_output)
        @print("histogram values: ", shared_histogram[tid], '\n')
        @inbounds histogram_output[input[tid]] += shared_histogram[input[tid]]
    end

    @synchronize()
end

function histogram_tag(input; numcores = 4, numthreads = 256)
    # I don't know how maximum is optimized on the GPU
    histogram_output = zeros(Int64, maximum(input))
    if isa(input, Array)
        event = histogram_tag!(histogram_output, input;
                               numcores = numcores, numthreads = numthreads)
        wait(event)
    else
        event = histogram_tag!(CuArray(histogram_output), input;
                               numcores = numcores, numthreads = numthreads)
        wait(event)
    end
    return histogram_output
end

function histogram_tag!(histogram_output, input;
                        numcores = 4, numthreads = 256)

    if isa(input, Array)
        kernel! = histogram_tag_kernel!(CPU(), numcores)
    else
        kernel! = histogram_tag_kernel!(CUDADevice(), numthreads)
    end

    kernel!(histogram_output, input, ndrange=size(input))
end

#=
@testset "histogram tests" begin

    a = [Int64(rand(1:128)) for i = 1:1000]
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
