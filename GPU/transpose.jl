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

function tiled_gpu_copy(a_in, a_out, tile_res)
    i = (blockIdx().x-1) * tile_res + threadIdx().x
    j = (blockIdx().y-1) * tile_res + threadIdx().y

    tile = @cuDynamicSharedMem(Float32, (tile_res,tile_res))

    for k = 1:blockDim().x:tile_res
        @inbounds tile[threadIdx().x + k*tile_res, threadIdx().y] = a_in[i+k, j*tile_res]
    end

    sync_threads()

    for k = 1:blockDim().x:tile_res
        @inbounds a_out[i+k, j*tile_res] = tile[threadIdx().x+k*tile_res, threadIdx().y]
    end

    return nothing

end

function main()

    #res = 8192
    # blocks must be defined based on tile size that can fit into shared mem
    tile_res = 32
    res = 128

    #a = round.(rand(Float32, (res, res))*100)
    a = ones(Float32, (res, res))
    d_a = CuArray(a)
    d_b = CuArray(zeros(Float64, res, res))

    println("Copy time is:")
    CuArrays.@time @cuda(threads = (tile_res, 8, 1),
                         blocks = (div(res,tile_res),div(res,tile_res),1),
                         shmem = sizeof(Float32)*tile_res*tile_res,
                         tiled_gpu_copy(d_a, d_b, tile_res))

#=
    println("Transpose time is:")
    CuArrays.@time @cuda(threads = (128, 1, 1),
                         blocks = (div(res,128),res,1),
                         naive_gpu_transpose(d_a, d_b))

    CuArrays.@time @cuda(threads = (128, 1, 1),
                         blocks = (div(res,128),res,1),
                         shmem = sizeof(Float32)*tile_res*tile_res),
                         tiled_gpu_transpose(d_a, d_b, res))
=#

    a = Array(d_a)
    b = Array(d_b)


    #if (a == transpose(b))
    if (a == b)
        println("Good job, man")
    else
        println("You failed. Sorry.")
    end

    return a .- b
end

main()
