using Plots
using FFTW

function find_nyquist_frequency(filebase::String, res::Int64)
    arr = Array{Float64}(undef, res)
    for i = 1:res
        for j = 1:res
             arr[j] = sin(j*i*2*pi/res)
        end
        plot(arr, ylim = (-1,1))
        savefig(filebase * string(lpad(string(i-1), 4, string(0))) * ".png")

        plot(abs.(fft(arr)))
        savefig(filebase *"FFT"* string(lpad(string(i-1), 4, string(0)))
                         * ".png")
        println(i)
    end
end

find_nyquist_frequency("images/nyfreq", 40)
