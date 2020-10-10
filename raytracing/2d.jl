# TODO: allow for different shapes

# note: light moves as a particular speed with respect to the medium it is
#       moving through, so...
#       n_2*v = n_1*l + (n_1*cos(theta_1) - n_2*cos(theta_2))*n
#       Other approximations: r = n_1/n_2, c = -n*l

using Plots
using LinearAlgebra

mutable struct Ray
    # Velocity vector
    v::Tuple{Float64, Float64}

    # Position vector
    p::Tuple{Float64, Float64}

    # Set of positions
    positions::Array{Tuple{Float64, Float64}}
end

mutable struct Mirror
    # Normal vector
    n::Tuple{Float64, Float64}

    # Position of mirror
    p::Tuple{Float64, Float64}

    # Size of mirror
    scale::Float64
end

function horizontal_mirror(p::Tuple{Float64, Float64}, scale)
    return Mirror((0.0, 1.0), p, scale)
end

function vertical_mirror(p::Tuple{Float64, Float64}, scale)
    return Mirror((1.0, 0.0), p, scale)
end

function is_behind(ray, mirror)
    if dot(ray.p.-mirror.p, mirror.n) >= 0
        return true
    else
        return false
    end
end

# note: for reflection, l_x -> l_x, but l_y -> -l_y
#       In this case, the y component of l = cos(theta)*n
#       so new vector: v = l + 2cos(theta)*n
function reflect!(ray, n)
    ray.v = ray.v .- 2*dot(ray.v, n).*n
end

# TODO: implement lens
#=
mutable struct lens
end
=#

function plot_rays(rays::Vector{Ray}, filename)
    plt = plot()
    for ray in rays
        plot!(plt, ray.positions)
    end

    savefig(plt, filename)
end

function step!(ray::Ray, dt)
    # TODO: implement refraction / reflection
    ray.p = ray.p .+ ray.v.*dt
end

function propagate!(rays::Vector{Ray}, mirrors::Vector{Mirror}, n, dt)

    for i = 1:n
        for j = 1:length(rays)
            for mirror in mirrors
                if is_behind(rays[j], mirror)
                    println(j, '\t', rays[j].v)
                    reflect!(rays[j], mirror.n)
                    println(j, '\t', rays[j].v)
                end
            end
            step!(rays[j], dt)
            rays[j].positions[i] = rays[j].p
        end
    end
end

function parallel_propagate(ray_num, n)

    rays = [Ray((sqrt(0.5), sqrt(0.5)),
            (float(i), 0.0),
            Array{Tuple{Float64, Float64}}(undef, n)) for i = 1:ray_num]
    mirrors = [horizontal_mirror((5.0,5.0), 5.0)]

    propagate!(rays, mirrors, n, 1.0)

    return rays
end
