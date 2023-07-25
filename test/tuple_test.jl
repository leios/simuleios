using KernelAbstractions, CUDA, CUDAKernels

function U(args...) end

# use repr to go from expr -> string
# think about recursive unions (barnsley + sierpinski)
function generate_H(expr)
    fnum = length(expr.args)-1
    fx_string = "function H(val, fid)\n"
    for i = 1:fnum
        temp_string = ""
        if i == 1
            f_str = repr(expr.args[i+1])[2:end]
            println(f_str)
            temp_string = "if fid == "*string(i)*" "*f_str*"(val)\n"
        else
            f_str = repr(expr.args[i+1])[2:end]
            println(f_str)
            temp_string = "elseif fid == "*string(i)*" "*f_str*"(val)\n"
        end
        fx_string *= temp_string
    end

    fx_string *= "else error('Function not found!')\n"
    fx_string *= "end\n"
    fx_string *= "end"

    H = Meta.parse(replace(fx_string, "'" => '"'))

    println(fx_string)
    println(H)

    return eval(H)

end

@kernel function f_test_kernel!(input, H, fnum)
    tid = @index(Global, Linear)

    #@print(N)

    meh = tid%fnum+1

    input[tid] = H(tid, meh)
end

function test!(input, H, fnum; numcores = 4, numthreads = 256)

    if isa(input, Array)
        kernel! = f_test_kernel!(CPU(), numcores)
    else
        kernel! = f_test_kernel!(CUDADevice(), numthreads)
    end

    kernel!(input, H, fnum, ndrange=size(input)[1])

end

@inline function f(x)
    x+1
end
@inline g(x) = x+2
@inline h(x) = x+3

fxs = :(U(f,g,h))

H = generate_H(fxs)

input = zeros(1024)
tuple_thingy = (f,g,h)
tuple_size = 3

wait(test!(input, H, 3))

d_input = CuArray(zeros(1024))
wait(test!(d_input, H,3))
