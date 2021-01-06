using FFTW
using Plots
using LinearAlgebra

import Plots: default
default(show=false, reuse=true)

ENV["GKSwstype"]="nul"

function signal_plot(i, signal1, signal2, out; filebase = "out",
                     out_name = "convolution", scale = 1.0)
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
    plt = plot(real(signal1); label = "signal (f)",
               ylabel = "Signal and Filter values", dpi=200, legend=:topleft,
               ylims=(0,1), size=(600,400), right_margin=12mm)
    plt = plot!(plt, shift_signal; label = "filter (g)")
    plt = plot!(plt, zeros(n1); label = "area under fg",
                ribbon=(zeros(n1),conv_sum))

    plt = twinx()
    plt = plot!(plt, real(out); label = out_name,
                linewidth=2, linestyle=:dash, ylabel = "Convolution output", 
                grid=:off, ylims = (0,scale+0.1), linecolor=:purple, 
                legend=:topright)
    savefig(filebase*lpad(string(i),5,string(0))*".png")

    #plt = plot(real(out), ylim=(0,scale); label = out_name, dpi=200)
    #savefig(filebase*"_conv_"*lpad(string(i),5,string(0))*".png")
end

function conv_fft(signal1::Array{Float64,1},
                  signal2::Array{Float64,1})
    return ifft(fft(signal1).*fft(signal2))
end

function conv_lin(signal1::Array{Float64,1},
                  signal2::Array{Float64,1};
                  norm_factor = 1, plot = false)
    n = length(signal1)
    out = zeros(n)
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
        out[i] = rsum
        if plot
            signal_plot(i, signal1, signal2, out; scale = norm_factor)
        end
        rsum = 0
    end

    return out
end

function main()
    #x = [exp(-((i-50)/100)^2/.01) for i = 1:100]
    #y = [exp(-((i-50)/100)^2/.01) for i = 1:100]
    x = zeros(100)
    x[40:60] .= 1.0 + 0im
    y = [(float(i))/20 for i = 20:-1:0]
    #normalize!(x)
    #normalize!(y)
    out_conv = conv_lin(x,y, norm_factor=sum(x.*x)*0.5; plot=true)

    return x
end
