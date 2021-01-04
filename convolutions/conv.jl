using FFTW
using Plots
using LinearAlgebra

import Plots: default
default(show=false, reuse=true)

ENV["GKSwstype"]="nul"

function signal_plot(i, signal1, signal2, out; filebase = "out",
                     out_name = "convolution")
    println(i)
    n1 = length(signal1)
    n2 = length(signal2)
    if n2 < n1
        signal2 = [reverse(signal2);zeros(n1-n2)]
    elseif n1 > n2
        signal1 = [signal1;zeros(n2-n1)]
    end
    shift_signal = real([signal2[n1-i:n1-1];signal2[1:n1-i]])
    conv_sum = real(signal1).*shift_signal
    plt = plot(real(signal1); label = "signal (f)", dpi=200)
    plt = plot!(plt, shift_signal; label = "filter (g)")
    plt = plot!(plt, zeros(n1); label = "area under fg", ribbon=(zeros(n1),conv_sum))
    plt = plot!(plt, real(out); label = out_name)
    savefig(filebase*lpad(string(i),5,string(0))*".png")
end

function conv_fft(signal1::Array{Complex{Float64},1},
                  signal2::Array{Complex{Float64},1})
    return ifft(fft(signal1).*fft(signal2))
end

function conv_lin(signal1::Array{Complex{Float64},1},
                  signal2::Array{Complex{Float64},1};
                  norm_factor = 1, plot = false)
    n = length(signal1)
    out = zeros(Complex, n)
    rsum = 0

    for i = 1:n-1
        if i+length(signal2) <= length(signal1)
            rsum = sum(signal1[i:i+length(signal2)-1].*reverse(signal2))
        else
            offset = i+length(signal2)-length(signal1)
            println("offset is ", offset)
            rsum = sum(vcat(signal1[i:end], signal1[1:offset-1])
                       .*reverse(signal2))
        end
        out[i] = rsum / norm_factor
        if plot
            signal_plot(i, signal1, signal2, out)
        end
        rsum = 0
    end

    return out
end

function main()
    x = [exp(-((i-50)/100)^2/.01) + 0im for i = 1:100]
    x = zeros(Complex{Float64}, 100)
    x[40:60] .= 1.0 + 0im
    #y = [exp(-((i-50)/100)^2/.01) + 0im for i = 1:100]
    y = [float(i)+0im for i = 19:-1:0]
    println(typeof(y))
    normalize!(x)
    normalize!(y)
    out_conv = conv_lin(x,y, norm_factor=sum(x.*x); plot=true)

    return x
end
