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

    Particle(p::Vector{Float64}, prev_p::Vector{Float64}, vel::Vector{Float64}, 
             acc::Vector{Float64}, mass::Float64, radius::Float64) = 
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
    Node(p::Vector{Float64}, box_length::Float64, parent::Nullable{Node}, 
         children::Vector{Node}, com::Particle, p_vec::Vector{Particle}) =
        new(p, box_length, parent, children, com, p_vec)
end

#------------------------------------------------------------------------------#
# SUBROUTINES
#------------------------------------------------------------------------------@

# Function to create random distribution
function create_rand_dist(box_length::Float64, pnum::Int)
    # initialize p_vec
    p_vec = fill(Particle(zeros(3), zeros(3), zeros(3), zeros(3),1.0,1.0), 0)
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
    com = Particle(zeros(3), zeros(3), zeros(3), zeros(3),1.0,1.0)
    root = Node([0.0; 0.0; 0.0], box_length, Nullable{Node}(), 
                Array(Node, 0), com, p_vec)
    return root
end

# Function to create children in a given node
function make_octchild(curr::Node)
    node_length = curr.box_length * 0.5
    quarter_box = curr.box_length * 0.25

    # Defining null com
    com = Particle(zeros(3), zeros(3), zeros(3), zeros(3),1.0,1.0)

    # iterating through possible location for new children nodes
    # wrtten by laurensbl
    for k = -1:2:1
        for j = -1:2:1
            for i = -1:2:1
                n = 2 * k + j + (i+1)/2 + 3;
                push!(curr.children, Node([curr.p[1] + k * quarter_box;
                                      curr.p[2] + j * quarter_box;
                                      curr.p[3] + i * quarter_box],
                                      curr.box_length*.5, Nullable{Node}(curr), 
                                      Array(Node, 0), com, 
                                      fill(com,0)))
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
function divide_octree(curr::Node, box_threshold::Int64)
    make_octchild(curr)
    for i = 1:8
        for p in curr.p_vec
            if (in_box(curr.children[i], p))
                push!(curr.children[i].p_vec, p)
                curr.children[i].p += p.p * p.mass
                curr.children[i].com.mass += p.mass
            end
        end
        curr.children[i].com.p /= curr.children[i].com.mass
        if (length(curr.children[i].p_vec) > box_threshold)
            divide_octree(curr.children[i], box_threshold)
        end
    end
end

# Function to output particle positions
# Note: Don't forget to opena nd close file surrounding this function
function particle_output(curr::Node, file::IOStream)
    if (length(curr.p_vec) == 1)
        writedlm(file,curr.p_vec[1].p')
    end

    for child in curr.children
        particle_output(child, file)
    end
end

# Function to find acceleration of single particle
function RKsearch(curr::Node, part::Particle)
    d = -part.p
    inverse_r = 1/sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])

    if (inverse_r > 10.0)
        inverse_r = 10.0
    end
    if (inverse_r < -10.0)
        inverse_r = -10.0
    end

    part.acc += d * inverse_r * 10.0
    
end

# Verlet integration
function verlet(part::Particle, dt::Float64)
    temp = part.p
    part.p = 2 * part.p - part.prev_p + part.acc * dt * dt
    part.prev_p = temp
end

# Function to find acceleration of particles in Barnes-Hut tree
function force_integrate(root::Node, dt::Float64)

    # Going through all particles in p_vec to find a new acceleration
    for part in root.p_vec
        RKsearch(root, part)
    end

    # Going through all particles and updating position
    for part in root.p_vec
        verlet(part, dt)
    end
    
end

# Main function
function main()
    numsteps = 5
    p_output = open("pout_julia.dat", "w")

    vel = sqrt(10)
    dt = 0.01
    p_vec = fill(Particle([1.0; 0.0; 0.0],[cos(vel*dt); sin(vel*dt); 0.0],
                 [0.0; vel; 0.0],zeros(3),1E10,1.0), 1)
    println(p_vec[1].prev_p)

    root = make_octree(p_vec, 1000.0)
    divide_octree(root, 1)

    for i = 1:numsteps
        root = make_octree(p_vec, 1000.0)
        divide_octree(root, 1)
        if (i % 1 == 0 || (i-1) == 0)
            particle_output(root, p_output)
            write(p_output, "\n\n")
        end
        force_integrate(root, 0.01)
    end

    close(p_output)
end

@time main()
