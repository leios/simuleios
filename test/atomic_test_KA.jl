using KernelAbstractions, CUDA, CUDAKernels

@kernel function f_test_kernel!(output, input)
    tid = @index(Global, Linear)
    lid = @index(Local, Linear)

    @uniform warpsize = Int(32)

    @uniform gs = @groupsize()[1]
    @uniform N = length(output)

    shared_tile = @localmem Int (gs)

    @uniform max_element = 1
    @uniform min_element = 1
    while max_element <= N

        # Setting shared_tile to 0
        @inbounds shared_tile[lid] = 0
        @synchronize()

        max_element = min_element + gs
        if max_element > N + 1
            max_element = N+1
        end

        if tid >= min_element && tid < max_element
            stid = tid - min_element + 1
            atomic_add!(pointer(shared_tile, stid), Int(input[tid]))
        end

        @synchronize()

        if ((lid+min_element-1) <= N)
            atomic_add!(pointer(output, lid+min_element-1), shared_tile[lid])
        end

        min_element += gs

    end
end

function test!(output, input; numcores = 4, numthreads = 256)

    AT = Array
    if isa(input, Array)
        kernel! = f_test_kernel!(CPU(), numcores)
    else
        AT = CuArray
        kernel! = f_test_kernel!(CUDADevice(), numthreads)
    end
    kernel!(output, input,
            ndrange=size(input)[1])

end
