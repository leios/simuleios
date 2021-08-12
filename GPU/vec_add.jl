using CUDA, Test

function kernel_vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y
    c[i,j] = a[i,j] + b[i,j]
    return nothing
end

@testset "Basic GPU addition" begin

    res = 1024

    # CUDAdrv functionality: generate and upload data
    a = round.(rand(Float32, (1024, 1024)) * 100)
    b = round.(rand(Float32, (1024, 1024)) * 100)
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = similar(d_a)  # output array

    # run the kernel and fetch results
    # syntax: @cuda [kwargs...] kernel(args...)
    @cuda threads = (128, 1, 1) blocks = (div(res,128),res,1) kernel_vadd(d_a, d_b, d_c)

    # CUDAdrv functionality: download data
    # this synchronizes the device
    c = Array(d_c)
    a = Array(d_a)
    b = Array(d_b)

    @test a+b ≈ c
end

#GC.gc()
