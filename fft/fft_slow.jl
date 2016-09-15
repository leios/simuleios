#-------------fft_slow.jl------------------------------------------------------#
#
# Purpose: Implement a slow version of an fft, to see how it's done.
#          For this, we will be using the Cooley-Tukey algorithm
#
#------------------------------------------------------------------------------#

# Implementing the Cooley-Tukey Algorithm
function cooley_tukey(x)
    N = length(x)

    #println(N)

    if(N%2 !=0)
        println("Must be a power of 2!")
        exit(0)
    end
    if(N <= 2)
        println("DFT_slow")
        return DFT_slow(x)
    else
        x_even = cooley_tukey(x[1:2:N])
        x_odd = cooley_tukey(x[2:2:N])
        n = 0:N-1
        #n = n'
        half = div(N,2)
        factor = exp(-2im*pi*n/N)
        println(length(factor))
        println(length(x_even))
        println(length(x_odd))
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
