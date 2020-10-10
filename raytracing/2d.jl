# TODO: allow for different shapes and mirror sizes
# TODO: extend to variable refractive indices
# TODO: Error in refract function

using Plots
using LinearAlgebra

pyplot()

mutable struct Ray
    # Velocity vector
    v::Vector{Float64}

    # Position vector
    p::Vector{Float64}

    # Set of positions
    positions::Array{Float64}
end

mutable struct Lens
    # Lens position
    p::Vector{Float64}

    # Lens radius
    r::Float64

    # Index of refraction
    ior::Float64
end

function lens_normal_at(ray, lens)
     return normalize(ray.p .- lens.p)
end

function inside_of(ray, lens)
    x = lens.p[1] - ray.p[1]
    y = lens.p[2] - ray.p[2]
    if (x^2 + y^2 <= lens.r^2)
        return true
    else
        return false
    end
end

# note: light moves as a particular speed with respect to the medium it is
#       moving through, so...
#       n_2*v = n_1*l + (n_1*cos(theta_1) - n_2*cos(theta_2))*n
#       Other approximations: r = n_1/n_2, c = -n*l
function refract!(ray, lens)
    n = lens_normal_at(ray, lens)
    c = dot(-n, ray.v);
    d = 1.0 - lens.ior^2 * (1.0 - c^2);

    if (d < 0.0)
        return [0.0, 0.0];
    end

    ray.v = lens.ior * ray.v + (lens.ior * c - sqrt(d)) * n;
end

mutable struct Mirror
    # Normal vector
    n::Vector{Float64}

    # Position of mirror
    p::Vector{Float64}

    # Mirror size
    scale::Float64

    Mirror(in_n, in_p) = new(in_n, in_p, 2.5)
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
    ray.v .-= 2*dot(ray.v, n).*n
end

function draw_circle(p, r, res)
    return [x .+ (p[1], p[2]) for x in Plots.partialcircle(0, 2pi, res, r)]
end

function plot_rays(rays::Vector{Ray}, mirrors::Vector{Mirror},
                   lenses::Vector{Lens}, filename)
    plt = plot()
    for ray in rays
        plot!(plt, (ray.positions[:,1], ray.positions[:,2]); label = "ray")
    end

    for mirror in mirrors
        dir = [-mirror.n[2], mirror.n[1]]
        extents = zeros(2,2)
        extents[1,:] .= mirror.p .- mirror.scale * dir
        extents[2,:] .= mirror.p .+ mirror.scale * dir
        plot!(plt, (extents[:,1], extents[:,2]), label = "mirror")
    end 

    for lens in lenses
        circle = draw_circle(lens.p, lens.r, 100)
        plot!(circle, label="lens")
    end 

    savefig(plt, filename)
end

function step!(ray::Ray, dt)
    ray.p .+= .+ ray.v.*dt
end

function propagate!(rays::Vector{Ray}, mirrors::Vector{Mirror},
                    lenses::Vector{Lens}, n, dt)

    for i = 1:n
        for j = 1:length(rays)
            for mirror in mirrors
                if is_behind(rays[j], mirror)
                    reflect!(rays[j], mirror.n)
                end
            end
            for lens in lenses
                if inside_of(rays[j], lens)
                    refract!(rays[j], lens)
                end
            end
            step!(rays[j], dt)
            rays[j].positions[i,:] .= rays[j].p
        end
    end
end

function parallel_propagate(ray_num, n; filename="check.png", dt = 0.1)

    rays = [Ray([sqrt(0.5), sqrt(0.5)],
            [float(i), 0.1],
            zeros(n, 2)) for i = 1:ray_num]

    lenses = [Lens([15, 10], 5, 1.5)]
    mirrors = [Mirror([0.0, -1.0],[5.0, 0.0])]
#=
    mirrors = [Mirror([0.0, 1.0],[5.0, 5.0]),
               Mirror([0.0, -1.0],[5.0, 0.0]),
               Mirror([1.0, 0.0],[0.0, 2.5]),
               Mirror([-1.0, 0.0],[10, 2.5])]
=#

    propagate!(rays, mirrors, lenses, n, dt)

    plot_rays(rays, mirrors, lenses, filename)

    return rays
end
