using DelimitedFiles

# Create a set of affine transforms for function system
function create_sierpinski_set(A, B, C)
    return [[0.5 0 A[1]*0.5; 0 0.5 A[2]*0.5; 0 0 1],
            [0.5 0 B[1]*0.5; 0 0.5 B[2]*0.5; 0 0 1],
            [0.5 0 C[1]*0.5; 0 0.5 C[2]*0.5; 0 0 1]]
end

function create_barnsley_set()
    return [[0.0 0.0 0.0; 0.0 0.16 0.0; 0.0 0.0 1.0],
            [0.85 0.04 0.0; -0.04 0.85 1.60; 0.0 0.0 1.0],
            [0.20 -0.26 0.0; 0.23 0.22 1.60; 0.0 0.0 1.0],
            [-0.15 0.28 0.0; 0.26 0.24 0.44; 0.0 0.0 1.0]]
end

# This is a function to simulate a "chaos game"
function chaos_game(n::Int, hutchinson)

    # Initializing the output array and the initial point
    output_points = zeros(n,2)
    point = [rand(), rand(), 1]

    for i = 1:n
        output_points[i,:] .= point[1:2]
        point = rand(hutchinson)*point
    end

    return output_points

end

# This will generate a Sierpinski triangle with a chaos game of n points for an 
# initial triangle with three points on the vertices of an equilateral triangle:
#     A = (0.0, 0.0)
#     B = (0.5, sqrt(0.75))
#     C = (1.0, 0.0)
# It will output the file sierpinski.dat, which can be plotted after
#hutchinson = create_sierpinski_set([0.0, 0.0], [0.5, sqrt(0.75)], [1.0, 0.0])
hutchinson = create_barnsley_set()
output_points = chaos_game(1000, hutchinson)
writedlm("out.dat", output_points)
