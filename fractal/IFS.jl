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
    ret_point = zeros(6)
    ret_point[1] = point[1] * c
    ret_point[2] = point[2] * c
    ret_point[3] = color[1]
    ret_point[4] = color[2]
    ret_point[5] = color[3]
    ret_point[6] = color[4]

    return ret_point
end

function heart(point::Array{Float64, 1}, color::Array{Float64, 1})
    ret_point = zeros(6)
    r = sqrt(point[1]^2 + point[2]^2)
    theta = atan(point[2],point[1])
    ret_point[1] = r*(sin(r*theta))
    ret_point[2] = -r*(cos(r*theta))
    ret_point[3] = color[1]
    ret_point[4] = color[2]
    ret_point[5] = color[3]
    ret_point[6] = color[4]

    return ret_point
end

function sinusoidal(point::Array{Float64, 1}, color::Array{Float64, 1})
    ret_point = zeros(6)
    ret_point[1] = sin(point[1])
    ret_point[2] = cos(point[2])
    ret_point[3] = color[1]
    ret_point[4] = color[2]
    ret_point[5] = color[3]
    ret_point[6] = color[4]

    return ret_point
end

function swirl(point::Array{Float64, 1}, color::Array{Float64, 1})
    ret_point = zeros(6)
    r = sqrt(point[1]^2 + point[2]^2)
    ret_point[1] = point[1]*sin(r^2) - point[2]*cos(r^2)
    ret_point[2] = point[1]*cos(r^2) + point[2]*sin(r^2)
    ret_point[3] = color[1]
    ret_point[4] = color[2]
    ret_point[5] = color[3]
    ret_point[6] = color[4]

    return ret_point
end

function sierpensky_flame(point::Array{Float64, 1}, color::Array{Float64, 1},
                          chosen_function::Int64)
    if (chosen_function == 1)
        point[1:2] *= 0.5
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    elseif(chosen_function == 2)
        point[1:2] = [(point[1] - 1)/2, (point[2])/2]
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    else
        point[1:2] = [(point[1])/2, (point[2]-1)/2]
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    end

    return point
end

function barnsley_flame(point::Array{Float64, 1}, color::Array{Float64, 1},
                        chosen_function::Int64)
    if (chosen_function == 1)
        point[1:2] = [0 0; 0 0.16]*point[1:2]
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    elseif(chosen_function == 2)
        point[1:2] = [0.85 0.04; -0.04 0.85]*point[1:2] + [0, 1.60]
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    elseif(chosen_function == 3)
        point[1:2] = [0.2 -0.26; 0.23 0.22]*point[1:2] + [0, 1.60]
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    else
        point[1:2] = [-0.15 0.28; 0.26 0.24]*point[1:2] + [0, 0.44]
        point[3] = color[1]
        point[4] = color[2]
        point[5] = color[3]
        point[6] = color[4]
    end

    return point

end

function write_image(p::Array{Pixel,2})
    a = Array{RGB{N0f8},2}(undef,size(p,1), size(p,2))
    for i = 1:length(pixels)
        a[i] = RGB{N0f8}(pixels[i].color[1],
                         pixels[i].color[2], 
                         pixels[i].color[3])
    end
    save("fractal.png", a)
end

function define_filter(sum::Float64, filter_size::Int64, fudge::Int64)

    filter = zeros(filter_size, filter_size)
    for i = 1:filter_size
        x = -2.0 + 4.0 * i / filter_size
        for j = 1:filter_size
            y = -2.0 + 4.0 * j / filter_size
            sigma2 = Float64(fudge)/(sum+1.0)
            filter[i,j] = (1.0/sqrt(2*pi*sigma2))*exp((-x^2 -y^2)/(2*sigma2))
        end
    end
    #println(filter)

    return filter
end

# Here, we perform a convolution with a variable filter (of constant size, but
# with varying gaussian widths) and trim the edges later.
function density_estimation(p::Array{Pixel, 2}, filter_size::Int64,
                            fudge::Int64)
    p_temp = zeros(size(p)[1]+filter_size, size(p)[2]+filter_size)
    for i = 1:size(p_temp)[1]
        for j = 1:size(p_temp)[2]
            start_pts = [i-div(filter_size,2); j-div(filter_size,2)]
            end_pts = [i+div(filter_size,2)-1; j+div(filter_size,2)-1]
            if (start_pts[1] < 1)
                start_pts[1] = 1
            end
            if (start_pts[2] < 1)
                start_pts[2] = 1
            end
            if (end_pts[1] > size(p)[1])
                end_pts[1] = size(p)[1]
            end
            if (end_pts[2] > size(p)[2])
                end_pts[2] = size(p)[2]
            end

            sum = 0.0
            for l = start_pts[1]:end_pts[1]
                for m = start_pts[2]:end_pts[2]
                    sum += p[l,m].val
                end
            end

            filter = define_filter(sum, filter_size, fudge)
            element = 0
            i2 = 1
            for l = start_pts[1]:end_pts[1]
                j2 = 1
                for m = start_pts[2]:end_pts[2]
                    element += p[l,m].val * filter[i2,j2]
                    j2 += 1
                end
                i2 += 1
            end
            p_temp[i,j] = element
        end
    end 

    for i = 1:size(p)[1]
        for j = 1:size(p)[2]
            p[i,j].val = p_temp[i+div(filter_size,2), j+div(filter_size,2)]
        end
    end 
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

function flame(n::Int64, resx::Int64, resy::Int64, 
               scalex::Float64, scaley::Float64, gamma::Float64)
    element_x = rand()*scalex*0.5 - 1
    element_y = rand()*scaley*0.5 - 1
    point = [element_x, element_y, 0.0, 0.0, 0.0]

    pixels = [Pixel([0.0, 0.0, 0.0], 0.0) for i = 1:resx, j = 1:resy]

    pixel_file = open("pixels.dat","w")

    color = zeros(3)
    for i = 1:n
        rnd = rand()
        r = sqrt(point[1]^2 + point[2]^2)
        if (abs(r) < 0.0001)
            r = 0.0001
        end
#=
        if (rnd <= 0.01)
            point = barnsley_flame(point, [1.0, 1.0, 1.0, 1.0], 1)
        elseif(rnd > 0.01 && rnd <= 0.86)
            point = barnsley_flame(point, [1.0, 0, 0, 1.0], 2)
        elseif(rnd > 0.86 && rnd <= 0.93)
            point = barnsley_flame(point, [0, 1.0, 0, 1.0], 3)
        else
            point = barnsley_flame(point, [0, 0, 1.0, 1.0], 4)
        end
=#
        if (rnd <= 0.33)
            #point = sierpensky_flame(point, [0.5, 0.5, 0, 1], 1)
            #point = linear(point, 0.5, [1.0, 0.0, 0.0, 1])
            point = sinusoidal(point, [0.0, 0.0, 1.0, 1])
        elseif (rnd > 0.33 && rnd <= 0.66)
            #point = sierpensky_flame(point, [0, 0.5, 0.5, 1], 2)
            point = linear(point, 0.5, [1.0, 0.0, 0.0, 1])
        else
            #point = sierpensky_flame(point, [0.5, 0, 0.5, 1], 3)
            #point = linear(point, 0.5, [1.0, 0.0, 0.0, 1])
            point = swirl(point, [1.0, 1.0, 1.0, 1])
        end

        # Final function attempts
        final_point = linear(point, .2, [0.0, 1.0, 0.0, 1])
        #final_point = sinusoidal(final_point, [0.0, 0.0, 1.0])

        # blending the colors
        if (i > 20)
            xval = Int(floor((final_point[1]+scalex*0.5) / (scalex / resx)))
            yval = Int(floor((final_point[2]+scaley*0.5) / (scaley / resy)))
            if (xval < resx && yval < resy && xval > 0 && yval > 0)
                pixels[yval,xval].val += 1
                pixels[yval,xval].color = (color+point[3:5]*point[6])*0.5
                color = pixels[yval,xval].color
                pixels[yval,xval].color = (color+final_point[3:5]*point[6])*0.5
            end
        end 

    end

    # find maximum in a dumb way
    max = 0;
    for pixel in pixels
        if (pixel.val > 0)
            pixel.val = log10(pixel.val)
        end
        if pixel.val > max
            max = pixel.val
        end
    end

    for i = 1:resy-1
        index_string = ""
        for j = 1:resx-1
            index_string *= string(log10(pixels[i,j].val), '\t')
        end
        println(pixel_file, index_string)
    end
    close(pixel_file)

    # Setting log plot and changing brightness
    for pixel in pixels
        pixel.val /= max
        pixel.color = (pixel.color * pixel.val).^(1/gamma)
        for c in pixel.color
            if c > 1
                c = 1
            end
        end
    end

    #density_estimation(pixels, 50, 1)

    return pixels

end

#pixels = flame(100000000, 1000, 1000, 4., 4., 1.0)
pixels = flame(1000000, 1000, 1000, 0.75, .75, 2.0)
write_image(pixels)
#sierpensky(1000000)
#barnsley_fern(100000)
