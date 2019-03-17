using Images, ImageMagick

function write_image(pixels::Array{HSV, 2}, output_file::String)
    color_array = Array{RGB{N0f8},2}(undef, size(pixels)[1], size(pixels)[2])

    for i = 1:size(pixels)[1]
        for j = 1:size(pixels)[2]
            color_array[i,j] = RGB(pixels[i,j])
        end
    end
    
    save(output_file, color_array)
end

function f(x::Complex{Float64})
    r = sqrt(real(x)^2 + imag(x)^2)
    theta = atan(imag(x), real(x))
    complex_val = r*exp(theta*im)
    return complex_val
end

function f2(x::Complex{Float64})
    r = sqrt(real(x)^2 + imag(x)^2)
    theta = atan(imag(x), real(x)) 
    complex_val = (r*exp(theta*im))^2
    return complex_val
end

function poly(x::Complex{Float64})
    r = sqrt(real(x)^2 + imag(x)^2)
    theta = atan(imag(x), real(x)) 
    complex_val = r*exp(theta*im)
    inverse = 1/complex_val
    if (abs(complex_val) < 0.001)
        inverse = 1/(0.001 + (0.001)im)
    end
    x = complex_val
    return factorial(x)
end

function inverse(x::Complex{Float64})
    r = sqrt(real(x)^2 + imag(x)^2)
    theta = atan(imag(x), real(x))
    complex_val = r*exp(theta*im)
    if abs(complex_val) < 0.005
        complex_val = 0.004 + 0.003im
    end 
    return 1/complex_val
end

function draw_grid(pixel_color::HSV, z::Complex{Float64}, threshold::Float64)

    shade_real = (abs(sin(real(z)*pi))^threshold)
    shade_imag = (abs(sin(imag(z)*pi))^threshold)

    return HSV(pixel_color.h,
               pixel_color.s,
               pixel_color.v*shade_real*shade_imag)
end

function contour(pixel_color::HSV, z::Complex{Float64})
    factor = 1-((abs(z) % 1)*0.5)
    pixel_color = HSV(pixel_color.h, pixel_color.s, pixel_color.v*factor)
    return pixel_color
end

function simple_domain(res::Int64, xmax::Float64, alpha::Float64,
                       output_file::String, f, threshold)
    a = Array{HSV,2}(undef, res, res)
    for i = 1:res
        for j = 1:res
            x = -xmax + 2*i*xmax/res + (-xmax + 2*j*xmax/res)im
            z = f(x)
            magnitude = abs(z)
            argument = angle(z) + pi
            
            a[i,j] = HSV(argument/(2*pi)*360, 1-alpha^(magnitude), 1)
            a[i,j] = draw_grid(a[i,j], z, threshold)
            a[i,j] = contour(a[i,j], z)
        end 
    end 
    write_image(a, output_file)
end
