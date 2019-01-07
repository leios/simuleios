using Images, ImageMagick

struct Pixel
    r::Float64
    g::Float64
    b::Float64
end

function write_image(pixels::Array{Pixel, 2}, output_file::String)
    color_array = Array{RGB{N0f8},2}(undef, size(pixels)[1], size(pixels)[2])

    for i = 1:size(pixels)[1]
        for j = 1:size(pixels)[2]
            color_array[i,j] = RGB{N0f8}(pixels[i,j].r,
                                         pixels[i,j].g,
                                         pixels[i,j].b)
        end
    end

    save(output_file, color_array)
end

function simple_domain(res::Int64, xmax::Float64, output_file::String)
    a = Array{Pixel,2}(undef, res, res)
    for i = 1:res
        for j = 1:res
            x = -xmax + 2*i*xmax/res
            xi = -xmax + 2*j*xmax/res
            r = sqrt(x*x + xi*xi)
            theta = atan(xi,x)
            a[i,j] = Pixel(r / (2*xmax), 0, (theta+pi) / (2*pi))
        end 
    end 
    write_image(a, "check.png")
end
