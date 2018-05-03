#-------------IFS.jl-----------------------------------------------------------#
#
# Purpose: quick test of iterated function system fractals
#
#------------------------------------------------------------------------------#

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

    f = open("barnsley.dat", "w")

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
    point = [element_x element_y]

    pixels = zeros(resx,resy)

    #out_file = open("out.dat","w")
    pixel_file = open("pixels.dat","w")

    for i = 1:n
        rnd = rand()
        r = sqrt(point[1]^2 + point[2]^2)
        if (abs(r) < 0.0001)
            r = 0.0001
        end

        if (rnd <= 0.33)
            point[1] = sin(point[1])
            point[2] = cos(point[2])
        elseif (rnd > 0.33 && rnd <= 0.66)
            #point[1] = (point[1] - point[2]) * (point[1] + point[2]) / r
            #point[2] = 2 * point[1] * point[2] / r
            point[1] = point[1] *0.5
            point[2] = point[2] *0.5
            #point[1] = point[1] + 3*sin(tan(3*point[2]))
            #point[2] = point[2] + 3*sin(tan(3*point[1]))
        else
            point[1] = point[1]*sin(r^2) - point[2]*cos(r^2)
            point[2] = point[1]*cos(r^2) + point[2]*sin(r^2)
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
            pixels[yval,xval] += 1
        end

    end

    max = maximum(pixels)
    for i = 1:resy-1
        index_string = ""
        for j = 1:resx-1
            #index = j + i*resy
            index_string *= string(log10(pixels[i,j]*255/max), '\t')
        end
        println(pixel_file, index_string)
    end
    #close(out_file)
    close(pixel_file)

    return pixels

end

#flame(1000000, 1000, 1000)
#sierpensky(10000)
#barnsley_fern(100000)
