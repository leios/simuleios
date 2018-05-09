#-------------IFS.jl-----------------------------------------------------------#
#
# Purpose: quick test of iterated function system fractals
#
#------------------------------------------------------------------------------#

using Images

mutable struct Pixel
    color::Vector{Float64}
    val::Float64
end

# Functions for fractal flames
function linear(point::Array{Float64, 1}, c::Float64, color::Array{Float64, 1})
    point[1] = point[1] * c
    point[2] = point[2] * c
    point[3] = color[1]
    point[4] = color[2]
    point[5] = color[3]

    return point
end

function sinusoidal(point::Array{Float64, 1}, color::Array{Float64, 1})
    point[1] = sin(point[1])
    point[2] = cos(point[2])
    point[3] = color[1]
    point[4] = color[2]
    point[5] = color[3]

    return point
end

function swirl(point::Array{Float64, 1}, color::Array{Float64, 1})
    r = sqrt(point[1]^2 + point[2]^2)
    point[1] = point[1]*sin(r^2) - point[2]*cos(r^2)
    point[2] = point[1]*cos(r^2) + point[2]*sin(r^2)
    point[3] = color[1]
    point[4] = color[2]
    point[5] = color[3]

    return point
end

function write_image(p::Array{Pixel,2})
    a = Array{RGB{N0f8},2}(size(p,1), size(p,2))
    for i = 1:length(pixels)
        a[i] = RGB{N0f8}(pixels[i].color[1], pixels[i].color[2], 
                         pixels[i].color[3])
    end
    save("fractal.png", a)
end

function barnsley_fern(n::Int64)
    point = [0.0, 0.0]
    f = open("barnsley.dat", "w")
    for i = 1:n
        rnd = rand()
        if (rnd <= 0.01)
            point = [0 0; 0 0.16]*point
        elseif(rnd > 0.01 && rnd <= 0.86)
            point = [0.85 0.04; -0.04 0.85]*point + [0, 1.60]
        elseif(rnd > 0.86 && rnd <= 0.93)
            point = [0.2 -0.26; 0.23 0.22]*point + [0, 1.60]
        else
            point = [-0.15 0.28; 0.26 0.24]*point + [0, 0.44]
        end

        println(f, point[1], '\t', point[2])
    end
    close(f)
end

function sierpensky(n::Int64)
    element_x = rand()*2 - 1
    element_y = rand()*2 - 1
    point = [element_x element_y]

    f = open("sierpensky.dat", "w")

    for i = 1:n
        if (i > 20)
            println(f, point[1], '\t', point[2])
        end
        rnd = rand()

        if (rnd <= 0.33)
            point = 0.5*point
        elseif(rnd > 0.33 && rnd <= 0.66)
            point[1] = (point[1] - 1)/2
            point[2] = (point[2])/2
        else
            point[1] = (point[1])/2
            point[2] = (point[2]-1)/2
        end
    end

    close(f)
end

function flame(n::Int64, resx::Int64, resy::Int64)
    element_x = rand()*2 - 1
    element_y = rand()*2 - 1
    point = [element_x, element_y, 0.0, 0.0, 0.0]

    pixels = [Pixel([0.0, 0.0, 0.0], 0.0) for i = 1:resx, j = 1:resy]

    #out_file = open("out.dat","w")
    pixel_file = open("pixels.dat","w")

    color = zeros(3)
    for i = 1:n
        rnd = rand()
        r = sqrt(point[1]^2 + point[2]^2)
        if (abs(r) < 0.0001)
            r = 0.0001
        end

        if (rnd <= 0.33)
            point = sinusoidal(point, [0.0, 0.0, 1.0])
        elseif (rnd > 0.33 && rnd <= 0.66)
            point = linear(point, 0.5, [1.0, 0.0, 0.0])
            #point[1] = (point[1] - point[2]) * (point[1] + point[2]) / r
            #point[2] = 2 * point[1] * point[2] / r
            #point[1] = point[1] + 3*sin(tan(3*point[2]))
            #point[2] = point[2] + 3*sin(tan(3*point[1]))
        else
            point = swirl(point, [0.0, 1.0, 0.0])
        end

        # Final function attempts
        #point[1] = (point[1] - point[2]) * (point[1] + point[2]) / r
        #point[2] = 2 * point[1] * point[2] / r
        #point[1] /= r^2
        #point[2] /= r^2
        #point[1] *= 0.5
        #point[2] *= 0.5

#=
        if (point[1] <= -1)
            point[1] = -0.99
        elseif (point[1] >= 1)
            point[1] = 0.99
        end

        if (point[2] <= -1)
            point[2] = -0.99
        elseif (point[2] >= 1)
            point[2] = 0.99
        end
=#

        #println(out_file, point[1], '\t', point[2])
        xval = Int(floor((point[1]+2) / (4 / resx)))
        yval = Int(floor((point[2]+2) / (4 / resy)))
        if (xval < resx && yval < resy && xval > 0 && yval > 0)
            #pixel_index = Int(xval + resy*yval)
            #println(pixel_index, '\t', point[1], '\t', point[2])
            pixels[yval,xval].val += 1
            pixels[yval,xval].color += point[3:5]
        end

    end

    # find maximum in a dumb way
    max = 0;
    for pixel in pixels
        if (pixel.val > 0)
            pixel.color /= pixel.val
            pixel.val = log10(pixel.val)
        end
        if pixel.val > max
            max = pixel.val
        end
    end

    for i = 1:resy-1
        index_string = ""
        for j = 1:resx-1
            #index = j + i*resy
            #index_string *= string(log10(pixels[i,j]*255/max), '\t')
            index_string *= string(log10(pixels[i,j].val), '\t')
        end
        println(pixel_file, index_string)
    end
    #close(out_file)
    close(pixel_file)

    for pixel in pixels
        pixel.val /= max
        if (pixel.val > 0.0000001)
            pixel.color = pixel.color * pixel.val
        end
    end

    return pixels

end

pixels = flame(10000000, 1000, 1000)
write_image(pixels)
#sierpensky(1000000)
#barnsley_fern(100000)
