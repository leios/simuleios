using CUDAnative, CuArrays, BenchmarkTools, StaticArrays

function tiled_gpu_transpose(a_in, a_out, tile_res, dim1, dim2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds cartesian_in = CartesianIndices(a_in)[i]

    tile = @cuDynamicSharedMem(Float32, tile_res)

    @inbounds tile[threadIdx().x] = a_in[i]

    sync_threads()

    # Changing index order for transpose.
    # Note: the only way I can see to do this is to transfer it into an array
    #       for a swap, and then transfer back
    tmp_idx = MVector(cartesian_in.I)
    @inbounds tmp_idx[dim1], tmp_idx[dim2] = tmp_idx[dim2], tmp_idx[dim1]
    cartesian_out = CartesianIndex(tmp_idx.data)

    @inbounds a_out[cartesian_out] = tile[threadIdx().x]

    return nothing
end

function naive_gpu_transpose(a_in, a_out, dim1, dim2)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds cartesian_in = CartesianIndices(a_in)[i]

    # Changing index order for transpose.
    # Note: the only way I can see to do this is to transfer it into an array
    #       for a swap, and then transfer back
    tmp_idx = MVector(cartesian_in.I)
    @inbounds tmp_idx[dim1], tmp_idx[dim2] = tmp_idx[dim2], tmp_idx[dim1]
    cartesian_out = CartesianIndex(tmp_idx.data)

    @inbounds a_out[cartesian_in] = a_in[cartesian_out]
    return nothing
end

function gpu_copy(a_in, a_out)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    @inbounds a_out[i] = a_in[i]
    return nothing
end

function tiled_gpu_copy(a_in, a_out, tile_res)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    tile = @cuDynamicSharedMem(Float32, tile_res)

    @inbounds tile[threadIdx().x] = a_in[i]

    sync_threads()

    @inbounds a_out[i] = tile[threadIdx().x]

    return nothing
end

function square_tiled_gpu_copy(a_in, a_out, tile_res)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds cartesian_in = CartesianIndices(a_in)[i]

    tile = @cuDynamicSharedMem(Float32, (sqrt(tile_res), sqrt(tile_res)))

    @inbounds tile[threadIdx().x] = a_in[i]

    sync_threads()

    @inbounds a_out[i] = tile[threadIdx().x]

    return nothing
end


function main()

    res = 8192
    # blocks must be defined based on tile size that can fit into shared mem
    tile_res = 1024
    #res = 1024

    a = round.(rand(Float32, (res, res))*100)
    #a = ones(Float32, (res, res))
    d_a = CuArray(a)
    d_b = CuArray(zeros(Float32, res, res))

    println("Copy time is:")
    CuArrays.@time @cuda(threads = (tile_res, 1, 1),
                         blocks = (div(res*res,tile_res),1,1),
                         gpu_copy(d_a, d_b))

    println("Transpose time is:")
    CuArrays.@time @cuda(threads = (tile_res, 1, 1),
                         blocks = (div(res*res,tile_res),1,1),
                         naive_gpu_transpose(d_a, d_b, 1, 2))

    println("Tiled copy time is:")
    CuArrays.@time @cuda(threads = (tile_res, 1, 1),
                         blocks = (div(res*res,tile_res),1,1),
                         shmem = sizeof(Float32)*tile_res,
                         tiled_gpu_copy(d_a, d_b, tile_res))

    println("Tiled transpose time is:")
    CuArrays.@time @cuda(threads = (tile_res, 1, 1),
                         blocks = (div(res*res,tile_res),1,1),
                         shmem = sizeof(Float32)*tile_res,
                         tiled_gpu_transpose(d_a, d_b, tile_res, 1, 2))

    a = Array(d_a)
    b = Array(d_b)

    if (a == transpose(b))
    #if (a == b)
        println("Good job, man")
    else
        println("You failed. Sorry.")
        return a .- b
    end

    return nothing
end

main()
