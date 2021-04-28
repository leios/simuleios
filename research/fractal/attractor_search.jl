# TODO: Fix image birghtness when writing images out to file

using DelimitedFiles
using LinearAlgebra
using Images
using ImageIO

# Function to output a set of points as an image
function create_image_array(points, ranges, res; point_offset = 10)
    output_array = zeros(res)

    range_diffs = [ranges[1,2] - ranges[1,1], ranges[2,2] - ranges[2,1]]

    # bin the pixels
    for i = point_offset:size(points)[1]

        yval = floor(Int, (points[i,1] - ranges[2,1]) /
                          (range_diffs[2]/ res[2]))
        xval = floor(Int, (points[i,2] - ranges[1,1]) /
                          (range_diffs[1] / res[1]))

        if (xval < res[1] && yval < res[2] && xval > 0 && yval > 0)
            output_array[xval,yval] += 1.0
        end 
    end

    if maximum(output_array) > 0
        return normalize(output_array)
    else
        return zeros(size(output_array))
    end

end

function write_image(points, ranges, res, filename; point_offset = 10)
    img_array = create_image_array(points, ranges, res;
                                   point_offset = point_offset)
    write_image(img_array, filename)
    
end

function write_image(img_array, filename)
    save(filename, reverse(img_array / maximum(img_array),dims=1))
end

# Create a set of affine transforms for function system
function create_sierpinski_set(A, B, C)
    return [[[0.5 0 A[1]*0.5; 0 0.5 A[2]*0.5; 0 0 1], .33],
            [[0.5 0 B[1]*0.5; 0 0.5 B[2]*0.5; 0 0 1], 0.33],
            [[0.5 0 C[1]*0.5; 0 0.5 C[2]*0.5; 0 0 1], 0.34]]
end

function create_barnsley_set()
    return [[[0.0 0.0 0.0; 0.0 0.16 0.0; 0.0 0.0 1.0], 0.01],
            [[0.85 0.04 0.0; -0.04 0.85 1.60; 0.0 0.0 1.0], 0.85],
            [[0.20 -0.26 0.0; 0.23 0.22 1.60; 0.0 0.0 1.0], 0.07],
            [[-0.15 0.28 0.0; 0.26 0.24 0.44; 0.0 0.0 1.0], 0.07]]
end

function select_array(hutchinson)

    rnd = rand()
    if rnd < hutchinson[1][2]
        return hutchinson[1][1]
    end

    rnd_sum = hutchinson[1][2]

    for i = 2:length(hutchinson)
        if rnd > rnd_sum && rnd < rnd_sum + hutchinson[i][2]
            return hutchinson[i][1]
        end
        rnd_sum += hutchinson[i][2]
    end

    return hutchinson[end][2]
    
end

# This is a function to simulate a "chaos game"
function chaos_game(n::Int, hutchinson)

    # Initializing the output array and the initial point
    output_points = zeros(n,2)
    point = [rand(), rand(), 1]

    for i = 1:n
        output_points[i,:] .= point[1:2]
        point = select_array(hutchinson)*point
    end

    return output_points

end

function main()
    #hutchinson = create_sierpinski_set([0.0, 0.0], [0.5, sqrt(0.75)], [1.0, 0.0])
    hutchinson = create_barnsley_set()
    output_points = chaos_game(100000000, hutchinson)
    #println(output_points)
    writedlm("out.dat", output_points)
    return output_points
end
