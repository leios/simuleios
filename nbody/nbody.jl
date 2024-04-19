using Plots
using KernelAbstractions

abstract type AbstractParticle end;

# initialization

struct SimpleParticle{T} <: AbstractParticle
    position::T
    velocity::T
    acceleration::T
end

struct MassParticle{T} <: AbstractParticle
    position::T
    velocity::T
    acceleration::T
    mass::Float64
end

function create_n_rand_particles(n, dim)
    [SimpleParticle(Tuple(2*rand(dim) .- 1),
                    Tuple(2*rand(dim) .- 1),
                    Tuple(2*rand(dim) .- 1)) for i = 1:n]
end

function create_position_array(p::Vector{AP}) where AP <: AbstractParticle
    dims = length(p[1].position)
    arr = zeros(length(p), dims)
    for i = 1:length(p)
        for j = 1:dims
            arr[i,j] = p[i].position[j]
        end
    end
    return arr
end

# Actual nbody sim

function gravity(p1::AbstractParticle, p2::AbstractParticle; G = 1)
    r2 = sum((p2.position .- p1.position).^2)
    unit_vector = (p2.position .- p1.position) ./ (sqrt(r2))
    return (G / (r2+1)) .* unit_vector
end

function find_acceleration(p1::AbstractParticle, p2::AbstractParticle;
                           force_law = gravity)
    return force_law(p1, p2)
end

function kinematic(position, velocity, acceleration, dt)
    return position .+ velocity .* dt .+ acceleration .* (dt*dt)
end

function move_particle(p1, acceleration, dt; routine = kinematic)
    new_velocity = acceleration .* dt .+ p1.velocity

    new_position = routine(p1.position, new_velocity, acceleration, dt)
    SimpleParticle(Tuple(new_position),
                   Tuple(new_velocity),
                   Tuple(acceleration))
end

function move_particles!(particles, accelerations, dt)
    particles .= move_particle.(particles, accelerations, dt)
end

function find_accelerations!(accelerations, particles)
    backend = get_backend(particles)
    kernel! = find_accelerations_kernel!(backend, 256)
    kernel!(accelerations, particles; ndrange = length(particles))
end

@kernel function find_accelerations_kernel!(accelerations, particles)
    j = @index(Global, Linear)
    new_acceleration = accelerations[j]
    particle = particles[j]
    for k = 1:length(particles)
        if j != k
            new_acceleration = new_acceleration .+
                               find_acceleration(particle,
                                                 particles[k])
        end
    end
    accelerations[j] = new_acceleration
end

function nbody(n, n_steps; dt = 0.01, dim = 2,
               plot_steps = 10, output_images = true, ArrayType = Array)

    particles = ArrayType(create_n_rand_particles(n, dim))
    accelerations = ArrayType([Tuple(zeros(dim)) for i = 1:n])

    for i = 1:n_steps
       accelerations .= (ntuple(_ -> 0, dim),)
       find_accelerations!(accelerations, particles)
       move_particles!(particles, accelerations, dt)

       if output_images
           if i%plot_steps == 0
               arr = create_position_array(Array(particles))
               plt = Plots.scatter(arr[:,1], arr[:,2];
                                   xlims = (-2, 2), ylims = (-2, 2))
               filename = "out"*lpad(i, 5, "0")*".png"
               savefig(plt, filename)
           end
       end
  
    end

    particles
end
