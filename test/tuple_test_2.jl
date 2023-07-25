using KernelAbstractions, CUDA, CUDAKernels

@generated function stable_nonsense(fns, idx, val)
    N = length(fns.parameters)
    quote
        Base.Cartesian.@nif $N d->d==idx d-> return fns[d](val)
    end
end

@kernel function f_test_kernel!(input, tuple_thingy::NTuple{N,Any}) where N
    tid = @index(Global, Linear)

    meh = tid%N+1

    input[tid] = stable_nonsense(tuple_thingy, meh, tid)
end

function test!(input, tuple_thingy::NTuple{N,Any}; numcores = 4, numthreads = 256) where N

    if isa(input, Array)
        kernel! = f_test_kernel!(CPU(), numcores)
    else
        kernel! = f_test_kernel!(CUDADevice(), numthreads)
    end

    kernel!(input, tuple_thingy, ndrange=size(input)[1])

end

function f(x)
    x+1
end
g(x) = x+2
h(x) = x+3

input = zeros(1024)
tuple_thingy = (f,g,h)
tuple_size = 3

wait(test!(input, tuple_thingy))

d_input = CuArray(zeros(1024))
wait(test!(d_input, tuple_thingy))
