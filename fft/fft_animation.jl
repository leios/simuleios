using Plots
using FFTW

function norm(a)
    sum = 0
    for i = 1:length(a)
        sum += a[i]
    end

    a=a/sum
    return a
end

function plot_fft(res, timesteps, max_freq)
    for i = 1:timesteps
        a = []
        if (i-1 <= div(timesteps,4))
            freq = (2+max_freq*(i-1)/(timesteps/4))
            a = [sin(freq*pi*Float64(j)/res) for j = 1:res]
        elseif (i-1 > div(timesteps,4) && i-1 <= div(timesteps,2))
            freq = (2+max_freq*(timesteps/2-(i-1))/(timesteps/4))
            a = [sin(freq*pi*Float64(j)/res) for j = 1:res]
        elseif (i-1 > div(timesteps,2) && i-1 < 3*div(timesteps,4))
            freq = (2 + max_freq*((i-1)-timesteps/2)/(timesteps/4))
            a = [sin(freq*pi*Float64(j)/res) for j = 1:res]
            b = [sin(2*freq*pi*Float64(j)/res) for j = 1:res]
            c = [sin(4*freq*pi*Float64(j)/res) for j = 1:res]
            a = a+b+c
        else
            freq = (2 + max_freq*(timesteps - (i-1))/(timesteps/4))
            a = [sin(freq*pi*Float64(j)/res) for j = 1:res]
            b = [sin(2*freq*pi*Float64(j)/res) for j = 1:res]
            c = [sin(4*freq*pi*Float64(j)/res) for j = 1:res]
            a = a+b+c
        end
        plot(a, label="Time Domain", legend = :top, linewidth = 2)
        savefig("images/time_" * string(lpad(i-1, 4, "0")) * ".png")

        b = abs2.(fft(a))
        b = norm(b)
        plot((b)[1:div(res,4)], label="Frequency Domain",
             legend = :top, linewidth = 2)
        savefig("images/freq_" * string(lpad(i-1, 4, "0")) * ".png")
    end
end

plot_fft(2000, 80, 40)
