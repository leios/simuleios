using CUDAnative, CUDAdrv, CuArrays, Test

function vec_sum(a, b, res)
    tid = threadIdx().x
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    tile = @cuDynamicSharedMem(Float64, res)

    tile[tid] = a[i]

    sync_threads()

    s = 1
    while s < blockDim().x
        if (tid % (2*s) == 1)
            tile[tid] += tile[tid+s]
        end
        sync_threads()
        s *= 2
    end 

    if tid == 1
        b[blockIdx().x] = tile[1]
    end 
    return nothing
end

function main()
    res = 32;

    a = [2.0 for i = 1:res]
    d_a = CuArray(a)
    d_a_out = similar(d_a)

    @cuda threads = (res, 1, 1) blocks = (1, 1, 1) shmem = sizeof(Float64)*res vec_sum(d_a, d_a_out, res)

    a_out = Array(d_a_out)
    a_check = Array(d_a)

    @test a_out[1] ≈ sum(a)
end

main()
