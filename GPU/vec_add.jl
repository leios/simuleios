using CUDAnative, CUDAdrv
using Test

function kernel_vadd(a, b, c, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if (i < n+1)
        c[i] = a[i] + b[i]
    end

    return nothing
end

# ERROR: Julia keeps memory allocated on GPU after running in REPL
function main()
    dev = CuDevice(0)
    ctx = CuContext(dev)

    # Generating some data
    len = 512
    a = rand(Int, len)
    b = rand(Int, len)

    # allocate and upload data to GPU
    d_a = CuArray(a)
    d_b = CuArray(b)
    d_c = similar(d_a)

    @cuda threads = (len, 1, 1) blocks = (1,1,1) kernel_vadd(d_a,d_b,d_c,len)
    c = Array(d_c)

    @test c == a+b

    destroy!(ctx)
end

main()
