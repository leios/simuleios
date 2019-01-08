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

function simple_domain(res::Int64, xmax::Float64, alpha::Float64,
                       output_file::String)
    a = Array{HSV,2}(undef, res, res)
    for i = 1:res
        for j = 1:res
            x = -xmax + 2*i*xmax/res
            xi = -xmax + 2*j*xmax/res
            r = sqrt(x*x + xi*xi)
            theta = atan(xi,x)
            a[i,j] = HSV(((theta+pi) / (2*pi))*360, 1-alpha^r, 1)
        end 
    end 
    write_image(a, output_file)
end
