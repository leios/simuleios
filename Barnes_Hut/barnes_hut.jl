#-------------barnes_hut.jl----------------------------------------------------#
#
# Purpose: To implement the Barnes-Hut galaxy simulator in julia
#
#------------------------------------------------------------------------------#

# Type for particles later to be used with Verlet integration
type Particle
    p::Vector{Float64}
    prev_p::Vector{Float64}
    vel::Vector{Float64}
    acc::Vector{Float64}
    mass::Float64
    radius::Float64

    Particle(p, prev_p, vel, acc, mass, radius) = 
        new(p, prev_p, vel, acc, mass, radius)
end

# type for node, to be used as an octree
type Node
    p::Vector{Float64}
    box_length::Float64
    parent::Nullable{Node}
    children::Vector{Node}
    com::Particle
    p_vec::Vector{Particle}
    Node(p, box_length, parent, children, com, p_vec) =
        new(p, box_length, parent, children, com, p_vec)
end

#------------------------------------------------------------------------------#
# SUBROUTINES
#------------------------------------------------------------------------------@

# Function to create random distribution
function create_rand_dist(box_length::Float64, pnum::Int)
    # initialize p_vec
    p_vec = fill(Particle([0],[0],[0],[0],1,1), 0)
    for i = 1:pnum
        push!(p_vec, Particle(rand(3)*box_length-(box_length * 0.5),
                              rand(3)*box_length-(box_length * 0.5),
                              rand(3)*box_length-(box_length * 0.5),
                              rand(3)*box_length-(box_length * 0.5),
                              1, 1))
    end
    return p_vec
end

# Function to initialize root node of octree
function make_octree(p_vec::Vector{Particle}, box_length::Float64)
    com = Particle([0],[0],[0],[0],1,1)
    root = Node([0; 0; 0], box_length, Nullable{Node}, 
                Array(Node, 0), com, p_vec)
    return root
end

# Function to create children in a given node
function make_octchild(curr::Node)
    node_length = curr.box_length * 0.5
    quarter_box = curr.box_length * 0.25

    # Defining null com
    com = Particle([0],[0],[0],[0],1,1)

    # iterating through possible location for new children nodes
    # wrtten by laurensbl
    for k = -1:2:1
        for j = -1:2:1
            for i = -1:2:1
                n = 2 * k + j + (i+1)/2 + 3;
                push!(curr.children, [curr.p[1] + k * quarter_box;
                                      curr.p[2] + j * quarter_box;
                                      curr.p[3] + i * quarter_box],
                                      curr.box_length, curr, 
                                      Array(Node, 0), com, 
                                      fill(com,0))
            end
        end
    end
end

# Function to check if we are in a box
function in_box(curr::Node, p::Particle)
    half_box = curr.box_length * 0.5
    return p.p[1] >= curr.p[1] - half_box &&
           p.p[1] <  curr.p[1] + half_box &&
           p.p[2] >= curr.p[2] - half_box &&
           p.p[2] <  curr.p[2] + half_box &&
           p.p[3] >= curr.p[3] - half_box &&
           p.p[3] <  curr.p[3] + half_box

end

# Function to divide root node onto octree
function divide_octree(curr::Node, box_threshold::Float64)
    make_octchild(curr)
    for i = 1:8
        for p in curr.p_vec
            if (in_box(curr.children[i], p))
                push!(curr.children[i].p_vec, p)
                curr.children[i].p += p.p * p.mass
                curr.children[i].mass += p.mass
            end
        end
        curr.children[i].com.p /= curr.children[i].com.mass
        if (length(curr.children[i].p_vec) > box_threshold)
            divide_octree(curr.children[i], box_threshold)
        end
    end
end

