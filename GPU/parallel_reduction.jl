using CUDAnative, CUDAdrv, CuArrays, Test

function vec_sum(a, b)
    tid = threadIdx().x

    s = 1
    while s < blockDim().x
        if (tid % (2*s) == 1)
            a[tid] += a[tid+s]
        end
        sync_threads()
        s *= 2
    end 

    if tid == 1
        b[blockIdx().x] = a[1]
    end 
    return nothing
end

function main()
    res = 128;

    #a = round.(rand(Float64, res)*100)
    a = [2 for i = 1:res]
    d_a = CuArray(a)
    d_a_out = similar(d_a)

    @cuda threads = (128, 1, 1) blocks = (1, 1, 1) vec_sum(d_a, d_a_out)

    a_out = Array(d_a_out)
    a_check = Array(d_a)

    @test a_out[1] ≈ sum(a)
end

main()
