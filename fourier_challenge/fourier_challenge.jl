using Plots
using ImageMagick
using FileIO
using Images
using ImageView
using FFTW

function convert_to_float(a::RGB)
    a = float(a)
    return (a.r + a.g + a.b)/3.0
end

function convert_to_float(a::RGBA)
    a = float(a)
    return (a.r + a.g + a.b + a.alpha)*0.25
end

function convert_to_rgba(r, i)
    return RGBA(r, 0, i, 1)
end

function normalize!(a)
    factor = 1/(sum(abs2.(a)))
    a .*= factor
end

function square_animation()
    x = zeros(100)
    x[1:5] .= 1
    x[end-5:end] .= 1
    for i = 1:length(x)-11
        x[i+5] = 1
        x[i] = 0
        if i < 6
            x[end-6+i] = 0
        end
        filename = "output/time"*lpad(i, 5, "0")*".png"
        time_plot = plot(x)
        savefig(time_plot, filename)

        y = fftshift(fft(x))
        filename = "output/frequency"*lpad(i, 5, "0")*".png"
        freq_plot = plot(real(y))
        freq_plot = plot!(freq_plot, imag(y))
        savefig(freq_plot, filename)

    end
end

function fourier_challenge(input_filename::String, output_filename::String)
    img = load(input_filename)
    img_real = float(red.(img))
    img_imag = float(blue.(img))

    array = img_real + im*img_imag
    array = fftshift(ifft(array))
    #array = ifft(array)
    array = abs2.(array)
    normalize!(array)
    img_final = convert_to_rgba.(real(array), imag(array))
    save(output_filename, map(clamp01nan, img_final))
end
