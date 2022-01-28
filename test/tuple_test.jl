using KernelAbstractions, CUDA, CUDAKernels

macro nif(N, condition, operation...)
    # Handle the final "else"
    ex = esc(Base.Cartesian.inlineanonymous(length(operation) > 1 ? operation[2] : operation[1] , N))
    # Make the nested if statements
    for i = N-1:-1:1
        ex = Expr(:if, esc(Base.Cartesian.inlineanonymous(condition,i)), esc(Base.Cartesian.inlineanonymous(operation[1],i)), ex)
    end
    ex
end

macro nif(N::Symbol, condition, operation...)
    # Handle the final "else"

    #ex = esc(Base.Cartesian.inlineanonymous(length(operation) > 1 ? operation[2] : operation[1] , eval(N)))
    # Make the nested if statements
    for i = eval(N)-1:-1:1
        ex = Expr(:if, esc(Base.Cartesian.inlineanonymous(condition,i)), esc(Base.Cartesian.inlineanonymous(operation[1],i)), ex)
    end
    ex
end

#function select_f(tuple_thingy, tuple_size

@kernel function f_test_kernel!(input, tuple_thingy::NTuple{N,Any}) where N
    tid = @index(Global, Linear)

    @print(N)

    meh = tid%tuple_size+1

    Base.Cartesian.@nif(N, d->meh == d, d->input[tid] = tuple_thingy[d](tid), d->@print(d,'\t',meh))
    #@nif(4, d->meh == d, d->input[tid] = tuple_thingy[d](tid), d->@print(d,'\t',meh))
    #Base.Cartesian.@nif(tuple_size, d->meh == d, d->println("OK: ", d,'\t',meh), d->println(d,'\t',meh))
#=
    for i = 1:tuple_size
        if meh == i
            input[tid] = tuple_thingy[i](tid)
        end
    end
=#
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
