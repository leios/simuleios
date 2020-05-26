using Images, ImageMagick

function write_image(pixels::Array{HSV, 2}, output_file::String)
    save(output_file, pixels)
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
    x = complex_val
    #return (x^2 - 1)*((x - 2 - 1im)^2) / (x^2 + 2 + 2im)
    return x^3 - 1
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

function write_domain(res::Int64, xmax::Float64, output::String, f)
    a = Array{Complex,2}(undef, res, res)

    file = open(output, "w")

    for i = 1:res
        for j = 1:res
            x = -xmax + 2*j*xmax/res + (-xmax + 2*i*xmax/res)im
            z = f(x)
            write(file, string(real(x)) *'\t' * string(imag(x)) *'\t' *
                        string(real(z)) * '\t' * string(imag(z)) * '\n')
        end
        write(file,'\n')
    end

    close(file)
end
