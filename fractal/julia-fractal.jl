#-------------julia-fractal.jl-------------------------------------------------#
#
# Purpose: This file tests the creation of the julia fractal in julia for fun
#
#------------------------------------------------------------------------------#

# We will need a few libraries for visualization
using PyPlot

# function to normalize fractal
function normalize(carr::Array{Float64,2})
    normalization_factor = sum(carr) / length(carr)
    carr /= normalization_factor
    return carr
end

# Function to write a fractal to a file (assuming square data)
function write_fractal(filename::String, res::Int64, fractal::Array{Float64})
    file = open(filename, "w")
    for i = 1:res
        line = ""
        for j = 1:res
            if j != res
                line *= string(fractal[i,j]) * '\t'
            else
                line *= string(fractal[i,j]) * '\n'
            end
        end
        write(file, line)
    end
    close(file)
end

# Function to calculate and output julia fractal to image
function create_fractal(cutoff::Float64, step::Float64, c_val::Float64, 
                        res::Int64, range::Float64)
    # Setting up the array to be used
    carr = zeros(res,res)

    # For now, let's just assume that the imaginary and real ranges are from 
    #     -range < x < range
    for i = 1:res
        for j = 1:res
            z = complex((i/res)*range - range/2.0,(j/res)*range - range/2.0)
            c = complex(0,c_val)
            n = 255
            while (abs(z) < cutoff && n > 5)
                z = z*z + c
                n -= 5
            end
            carr[i,j] = n/255.0;
        end
    end

    return normalize(carr)

end

# Fuction to scan through C values
function c_scan(cutoff::Float64, step::Float64, res::Int64,
                range::Float64, max_c::Float64, c_step::Float64)

    id = 0
    for i = 0:c_step:max_c
        println(i)
        carr = create_fractal(cutoff, step, i, res, range)
        filename = string("cdata/c_scan",
                          lpad(string(id), 5, string(0)), ".dat")
        write_fractal(filename, res, carr)
        id += 1
    end
end

# Function to zoom in on fractal
function fractal_zoom(cutoff::Float64, step::Float64, c_val::Float64,
                      res::Int64, max_range::Float64, 
                      min_range::Float64)
    id = 0
    range = max_range
    while range > min_range
        println(range)
        carr = create_fractal(cutoff, step, c_val, res, range)
        filename = string("data/fractal_zoom",
                          lpad(string(id), 5, string(0)), ".dat")
        write_fractal(filename, res, carr)
        id += 1
        range -= range*0.05
    end
end

#fractal_zoom(10.0, 0.1, 1.0, 512, 1.0, 0.0001)

c_scan(10.,5., 512, 3.0, 1.0, 0.025)
