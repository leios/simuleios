using FFTW
using Plots
using Plots.PlotMeasures
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
    plt = plot(real(signal1); label = "signal (f(x))",
               ylabel = "Signal and Filter values", dpi=200, legend=:topleft,
               ylims=(0,1.3), size=(600,400), right_margin=12mm,
               xlabel = "x")
    plt = plot!(plt, shift_signal; label = "filter (g(x-t))")
    plt = plot!(plt, zeros(n1); label = "area under f(x)g(x-t)",
                ribbon=(zeros(n1),conv_sum))

    plt = twinx()
    plt = plot!(plt, real(out); label = out_name,
                linewidth=2, linestyle=:dash, ylabel = "Convolution output", 
                grid=:off, ylims = (0,scale*1.3), linecolor=:purple, 
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
                  norm_factor = 1, plot = false, frame_output = 1)
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
        if plot && i % frame_output == 0
            signal_plot(i, signal1, signal2, out; scale = norm_factor)
        end
        rsum = 0
    end

    return out
end

function main()
    #x = [exp(-((i-500)/1000)^2/.01) for i = 1:1000]
    #y = [exp(-((i-500)/1000)^2/.01) for i = 1:1000]
    x = zeros(1000)
    x[400:600] .= 1.0 + 0im
    y = [(float(i))/200 for i = 200:-1:0]
    #normalize!(x)
    #normalize!(y)
    out_conv = conv_lin(x,y, norm_factor=sum(x.*x)*0.5; plot=true,
                        frame_output = 10)

    return x
end
