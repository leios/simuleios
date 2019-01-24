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

function f(r::Float64, theta::Float64)
    return  r + theta*im
end

function f2(r::Float64, theta::Float64)
    arg = 2*theta
    while (arg > 2pi)
        arg -= 2*pi
    end
    return r*r + arg*im
end

function draw_grid(pixel_color::HSV, z::Complex{Float64}, threshold::Float64)
    z = real(z)*exp(imag(z)*im)
    if (abs(real(z))%1 < threshold || abs(imag(z))%1 < threshold)
        return HSV(0,0,0)
    else
        return pixel_color
    end
end

function simple_domain(res::Int64, xmax::Float64, alpha::Float64,
                       output_file::String, f, threshold)
    a = Array{HSV,2}(undef, res, res)
    for i = 1:res
        for j = 1:res
            x = -xmax + 2*i*xmax/res
            xi = -xmax + 2*j*xmax/res
            r = sqrt(x*x + xi*xi)
            theta = atan(xi,x) + pi
            z = f(r, theta)
            a[i,j] = HSV((imag(z) / (2*pi))*360, 1-alpha^(real(z)), 1)
            a[i,j] = draw_grid(a[i,j], z, threshold)
        end 
    end 
    write_image(a, output_file)
end
