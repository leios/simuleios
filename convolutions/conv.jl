using FFTW
using Plots

import Plots: default
default(show=false, reuse=true)

ENV["GKSwstype"]="nul"

function signal_plot(i, signal1, signal2, out; filebase = "out",
                     out_name = "convolution")
    println(i)
    n = length(signal1)
    shift_signal = real([signal2[i:n-1];signal2[1:i]])
    conv_sum = real(signal1).*shift_signal
    plt = plot(real(signal1); label = "signal")
    plt = plot!(plt, shift_signal; label = "filter")
    plt = plot!(plt, zeros(n); label = "area", ribbon=(zeros(n),conv_sum))
    plt = plot!(plt, real(out); label = out_name)
    savefig(filebase*lpad(string(i),5,string(0))*".png")
end

function is_norm(signal)
    if isapprox(sum(signal), 1.0)
        return true
    else
        return false
    end
end

function norm(signal)
    return signal ./ sum(signal)
end

# TODO: better name than temp_signal
function expectation_value(signal)
    temp_signal = signal
    if !is_norm(signal)
        temp_signal = norm(signal)
    end 
    expectation_value = 0
    for i = 1:length(signal)
        expectation_value += i*temp_signal[i]
    end
    return expectation_value
end

function covariance(signal1, signal2)
    return expectation_value((signal1 .- expectation_value(signal1)).*
                             (signal2 .- expectation_value(signal2)))
end

function conv_lin(signal1::Array{Complex{Float64},1},
                  signal2::Array{Complex{Float64},1})
    n = length(signal1)
    out = Array{Complex{Float64},1}(undef,n)
    rsum = 0

    # time domain
    for i = 1:n-1
        # inner domain between signals
        for j = 1:length(signal2)
            if (j < length(signal1))
                index = i-j
                if i-j < 1
                    index = length(signal2)+(i-j)
                end
                rsum += signal1[j] * signal2[(index)%(length(signal2))+1]
            end
        end
        out[i] = rsum
        rsum = 0
    end

    return out
end

function conv_fft(signal1::Array{Complex{Float64},1},
                  signal2::Array{Complex{Float64},1})
    return ifft(fft(signal1).*fft(signal2))
end

function conv_plot(signal1::Array{Complex{Float64},1},
                   signal2::Array{Complex{Float64},1}; norm_factor = 1)
    n = length(signal1)
    out = zeros(Complex, n)
    rsum = 0

    # time domain
    for i = 1:n-1
        # inner domain between signals
        for j = 1:length(signal2)
            if (j < length(signal1))
                index = j-i
                if j-i < 1
                    index = length(signal2)+(j-i)
                end
                rsum += signal1[j] * signal2[(index)%(length(signal2))+1]
            end
        end


        out[i] = rsum / norm_factor
        rsum = 0

        signal_plot(i, signal1, signal2, out)

    end

    return out
end

function corr_lin(signal1::Array{Complex{Float64},1},
                  signal2::Array{Complex{Float64},1})
    n = length(signal1)
    out = Array{Complex{Float64},1}(undef,n)
    rsum = 0

    for i = 1:n-1
        for j = 1:length(signal2)
            if j < length(signal1)
                rsum += conj(signal1[j]) * signal2[(j+i)%(length(signal2))+1]
            end
        end
        out[i] = rsum
        rsum = 0
    end

    return out
end

function corr_plot(signal1::Array{Complex{Float64},1},
                   signal2::Array{Complex{Float64},1}; norm_factor = 1)
    n = length(signal1)
    out = zeros(Complex, n)
    rsum = 0

    for i = 1:n-1
        for j = 1:length(signal2)
            if j < length(signal1)
                rsum += conj(signal1[j]) * signal2[(j+i)%(length(signal2))+1]
            end

        end

        out[i] = rsum / norm_factor
        rsum = 0

        signal_plot(i, signal1, signal2, out)

    end

    return out
end

function main()
    x = [exp(-((i-50)/100)^2/.01) + 0im for i = 1:100]
    out_conv = conv_plot(x,x, norm_factor=20)


#=
    x = zeros(Complex{Float64},100)
    x[40:60] .= 1

    y = copy(x)
    y[40:60] .= [float(1-i*0.05) for i = 0:20]

    # Make plots
    out_corr = corr_plot(x,y;norm_factor=20; filebase = "corr_out")
    out_conv = conv_plot(x,y, norm_factor=20)
=#
end
