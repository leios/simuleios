using KernelAbstractions, CUDA, CUDAKernels, Test

function null()
end

function configure(H)
    H(1)
    return nothing
end

function cuda_test!(input, H)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= length(input)
        input[i] = H(i)
    end

    return nothing
end

@kernel function f_test_kernel!(input, H)
    tid = @index(Global, Linear)

    input[tid] = H(tid)
end

function test!(input, H; numcores = 4, numthreads = 256)

    if isa(input, Array)
        kernel! = f_test_kernel!(CPU(), numcores)
    else
        kernel! = f_test_kernel!(CUDADevice(), numthreads)
    end

    kernel!(input, H, ndrange=size(input)[1])

end

@testset begin
    input = zeros(1024)

    @inline H(tid) = tid*0
    wait(test!(input,H))
    @test(sum(input) == 0)

    @inline H(tid) = tid*0 + 1
    wait(test!(input,H))
    @test(sum(input) == 1024)

    input = CuArray(zeros(1024))

    @inline H(tid) = tid*0
    wait(test!(input,H))
    @test(sum(input) == 0)

    @cuda threads = (128, 1, 1) blocks = (128,1,1) configure(H)
    #@cuda threads = (128, 1, 1) blocks = (128,1,1) cuda_test!(input, H)
    #@test(sum(input) == 0)

    @inline H(tid) = tid*0 + 1
    wait(test!(input,H))
    @test(sum(input) == 1024)
    @cuda threads = (128, 1, 1) blocks = (128,1,1) configure(H)

    #@cuda threads = (128, 1, 1) blocks = (128,1,1) cuda_test!(input, H)
    #@test(sum(input) == 1024)
end
