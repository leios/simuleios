using Test
using CUDA
using CUDAKernels
using KernelAbstractions

# Function to use as a baseline for CPU metrics
function create_histogram(input)
    histogram_output = zeros(Int, maximum(input))
    for i = 1:length(input)
        histogram_output[input[i]] += 1
    end
    return histogram_output
end

function tag_bin!(shared_histogram, input, shared_min, shared_max, tid, lid)
end

# This is for a 1D histogram, for n-dim binning, we need to think a bit more
# The problem here is that the @synchronize calls are only synchronizing across
#     blocks, while we need a way to combine all the shmem histograms together 
#     into a big global blob. This can be done by:
@kernel function histogram_tag_kernel!(histogram_output, input)
    tid = @index(Global, Linear)

    bin = input[tid]

    tagged = 0xffffffff
    while true
        # Storing the value in 27 bits and erasing the first 5
        val = histogram_output[bin] & 0x07ffffff

        # tagging first 5
        tagged = ((tid % 32 ) << 27) | (val+1)

        histogram_output[bin] = tagged
        @synchronize()
        histogram_output[bin] != tagged || break
    end

    histogram_output[bin] = histogram_output[bin] & 0x07ffffff

end

function histogram_tag(input; numcores = 4, numthreads = 256)
    # I don't know how maximum is optimized on the GPU
    if isa(input, Array)
        histogram_output = zeros(UInt32, maximum(input))
        event = histogram_tag!(histogram_output, input;
                               numcores = numcores, numthreads = numthreads)
        wait(event)
        return histogram_output
    else
        histogram_output = CuArray(zeros(UInt32, maximum(input)))
        event = histogram_tag!(histogram_output, input;
                               numcores = numcores, numthreads = numthreads)
        wait(event)
        return histogram_output
    end
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

    a = [UInt32(rand(1:128)) for i = 1:1000]
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
