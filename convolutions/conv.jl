using FFTW
using Plots

import Plots: default
default(show=false, reuse=true)

ENV["GKSwstype"]="nul"

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

        println(i)
        plt = plot(real(signal1))
        plt = plot!(plt, real([signal2[n-i:n];signal2[1:n-i]]))
        plt = plot!(plt, real(out))
        savefig("out_conv_"*lpad(string(i),5,string(0))*".png")

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

        println(i)
        plt = plot(real(signal1))
        plt = plot!(plt, real([signal2[i:n];signal2[1:i]]))
        plt = plot!(plt, real(out))
        savefig("out"*lpad(string(i),5,string(0))*".png")

    end

    return out
end

function main()

    x = zeros(Complex{Float64},100)
    x[40:60] .= 1

    y = copy(x)
    y[40:60] .= [float(1-i*0.05) for i = 0:20]

    # Make plots
    out_corr = corr_plot(x,y;norm_factor=20)
    out_conv = conv_plot(x,y, norm_factor=20)
end
