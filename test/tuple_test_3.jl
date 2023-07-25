using KernelAbstractions, CUDA, CUDAKernels

function U(args...)
end

@generated function stable_nonsense(H::Expr, idx, val)
    N = length(H.args)-1
    quote
        Base.Cartesian.@nif $N d->d==idx d-> return eval(H.args[d+1])(val)
    end
end

@kernel function f_test_kernel!(input, H)
    tid = @index(Global, Linear)

    N = length(H.args)-1
    meh = tid%N+1

    input[tid] = stable_nonsense(H, meh, tid)
end

function test!(input, H::Expr; numcores = 4, numthreads = 256)

    if isa(input, Array)
        kernel! = f_test_kernel!(CPU(), numcores)
    else
        kernel! = f_test_kernel!(CUDADevice(), numthreads)
    end

    kernel!(input, H, ndrange=size(input)[1])

end

function f(x)
    x+1
end
g(x) = x+2
h(x) = x+3

input = zeros(1024)
H = :(U(f,g,h))
tuple_size = 3

wait(test!(input, H))

d_input = CuArray(zeros(1024))
wait(test!(d_input, tuple_thingy))
