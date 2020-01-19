using CUDAnative, CuArrays, BenchmarkTools

function tiled_gpu_transpose(a_in, a_out, res)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    #tile = @cuDynamicSharedMem(Float64, res*res)
    #tile[i,j] = a_in[i,j]

    #sync_threads()

    #@inbounds a_out[i,j] = tile[i,j]
    @inbounds a_out[i,j] = a_in[j,i]
    return nothing

end

function naive_gpu_transpose(a_in, a_out)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    @inbounds a_out[i,j] = a_in[j,i]
    return nothing
end

function gpu_copy(a_in, a_out)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    @inbounds a_out[i,j] = a_in[i,j]
    return nothing
end

function main()

    res = 8192
    #res = 128

    a = round.(rand(Float32, (res, res))*100)
    d_a = CuArray(a)
    d_b = similar(d_a)

    println("Copy time is:")
    CuArrays.@time @cuda(threads = (128, 1, 1),
                         blocks = (div(res,128),res,1),
                         gpu_copy(d_a, d_b))

    println("Transpose time is:")
    CuArrays.@time @cuda(threads = (128, 1, 1),
                         blocks = (div(res,128),res,1),
                         naive_gpu_transpose(d_a, d_b))

#=
    CuArrays.@time @cuda(threads = (128, 1, 1),
                         blocks = (div(res,128),res,1),
                         shmem = sizeof(Float64*tile_res*tile_res),
                         tiled_gpu_transpose(d_a, d_b, res))
=#

    a = Array(d_a)
    b = Array(d_b)

    if (a == transpose(b))
        println("Good job, man")
    else
        println("You failed. Sorry.")
    end
end

main()
