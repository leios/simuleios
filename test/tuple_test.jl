#using KernelAbstractions, CUDA, CUDAKernels

function U(args...) end

# use repr to go from expr -> string
# think about recursive unions (barnsley + sierpinski)
function generate_H(expr)
    fnum = length(expr.args)-1
    fx_string = "function H(val, fid)\n"
    for i = 1:fnum
        temp_string = ""
        if i == 1
            temp_string = "if fid == "*string(i)*" eval("*repr(expr.args[i+1])*")(val)\n"
        else
            temp_string = "elseif fid == "*string(i)*" eval("*repr(expr.args[i+1])*")(val)\n"
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

#=
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
=#
