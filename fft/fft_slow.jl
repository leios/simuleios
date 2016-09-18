#-------------fft_slow.jl------------------------------------------------------#
#
# Purpose: Implement a slow version of an fft, to see how it's done.
#          For this, we will be using the Cooley-Tukey algorithm
#
#------------------------------------------------------------------------------#

using PyPlot

# Implementing the Cooley-Tukey Algorithm
function cooley_tukey(x)
    N = length(x)

    #println(N)

    if(N%2 !=0)
        #println("Must be a power of 2!")
        exit(0)
    end
    if(N <= 2)
        #println("DFT_slow")
        return DFT_slow(x)
    else
        x_even = cooley_tukey(x[1:2:N])
        x_odd = cooley_tukey(x[2:2:N])
        n = 0:N-1
        #n = n'
        half = div(N,2)
        factor = exp(-2im*pi*n/N)
        #println(length(factor))
        #println(length(x_even))
        #println(length(x_odd))
        return vcat(x_even + factor[1:half] .* x_odd,
                     x_even + factor[half+1:N] .* x_odd) 
    end
    
end

# A super slow DFT to start
function DFT_slow(x)
    N = length(x)

    # range is automatically a column vector, and we want a row vector
    n = 0:N-1
    n = n'
    k = n'
    M = exp(-2im * pi *k *n / N)
    return M * x
end

# function to greyscale image
function greyscale(img)
    a = zeros(width(img), height(img))
    for i = 1:width(img)
        for j = 1:height(img)
            a[i,j] = (img[i,j].r + img[i,j].b + img[i,j].g) / 3
        end
    end

    return a
end

# function to determine whether a point is inside of a decided norm
function in_norm(x, y, power)
    if (abs(x)^power + abs(y)^power)^(1/power) < 1
        return true
    else
        return false
    end
end

# function to create array of points to work with
function monte_carlo(res, power)
    a = []
    for i = 1:res
        x = rand() * 2 - 1
        y = rand() * 2 - 1
        if (in_norm(x, y, power))
            push!(a, [x,y])
        end
    end

    x = map(x -> x[1], a)
    y = map(y -> y[2], a)
    scatter(x,y)
    #return x,y
end
