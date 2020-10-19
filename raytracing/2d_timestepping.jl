# TODO: allow for different shapes and mirror sizes

using Plots
using LinearAlgebra

pyplot()

mutable struct Ray
    # Direction vector
    l::Vector{Float64}

    # Position vector
    p::Vector{Float64}

    # Current Index of Refraction
    ior::Float64
end

mutable struct Lens
    # Lens position
    p::Vector{Float64}

    # Lens radius
    r::Float64

    # Lens function
    f
end

function constant(p)
    return 1.5
end

function mag_index(p)
    return cos(sum(p[:].^2))
end

function inverse(p)
    mag = sum(p[:].^2)
    return 1/(mag)
end

function lens_normal_at(ray, lens)
    n = normalize(ray.p .- lens.p)
    if dot(-n, ray.l) < 0
        n *= -1
    end

    return n
end

function inside_of(ray::Ray, lens)
    return inside_of(ray.p, lens)
end

function inside_of(pos, lens)
    x = lens.p[1] - pos[1]
    y = lens.p[2] - pos[2]
    if (x^2 + y^2 <= lens.r^2)
        return true
    else
        return false
    end
end

function refract!(ray, n)
    c = dot(-n, ray.l);
    d = 1.0 - ior^2 * (1.0 - c^2);

    if (d < 0.0)
        return zeros(2);
    end

    ray.l = ior * ray.l + (ior * c - sqrt(d)) * n;
end

# note: light moves as a particular speed with respect to the medium it is
#       moving through, so...
#       n_2*v = n_1*l + (n_1*cos(theta_1) - n_2*cos(theta_2))*n
#       Other approximations: ior = n_1/n_2, c = -n*l
function refract!(ray, lens, ior)
    n = lens_normal_at(ray, lens)
    c = dot(-n, ray.l);
    d = 1.0 - ior^2 * (1.0 - c^2);

    if (d < 0.0)
        return zeros(2);
    end

    ray.l = ior * ray.l + (ior * c - sqrt(d)) * n;
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
    ray.l .-= 2*dot(ray.l, n).*n
end

function draw_circle(p, r, res)
    return [x .+ (p[1], p[2]) for x in Plots.partialcircle(0, 2pi, res, r)]
end

function plot_rays(positions, mirrors::Vector{Mirror},
                   lenses::Vector{Lens}, filename)
    plt = plot(background_color=:black, aspect_ratio=:equal)
    for i = 1:size(positions)[1]
        plot!(plt, (positions[i,:,1], positions[i,:,2]); label = "ray",
              linecolor=:white)
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
        plot!(circle; label="lens", linecolor=:lightblue)
    end 

    savefig(plt, filename)
end

function step!(ray::Ray, dt)
    ray.p .+= .+ ray.l.*dt/ray.ior
end

function propagate!(rays::Vector{Ray}, mirrors::Vector{Mirror},
                    lenses::Vector{Lens}, positions, n, dt)

    ior = 1.0
    for i = 1:n
        for j = 1:length(rays)
            for mirror in mirrors
                if is_behind(rays[j], mirror)
                    reflect!(rays[j], mirror.n)
                end
            end
            for lens in lenses
                if inside_of(rays[j], lens) && i > 2
                   ior = rays[j].ior/lens.f(rays[j].p .- lens.p)
                   rays[j].ior = lens.f(rays[j].p .- lens.p)
                   refract!(rays[j], lens, ior)
                elseif !inside_of(rays[j], lens) && i > 2 &&
                       rays[j].ior != 1
                    ior = rays[j].ior
                    rays[j].ior = 1
                    refract!(rays[j], lens, ior)
                end
                if rays[j].l == zeros(2)
                    reflect!(rays[j], lens_normal_at(rays[j], lens))
                end
            end
            step!(rays[j], dt)
            positions[j, i,:] .= rays[j].p
        end
    end
end

function parallel_propagate(ray_num, n; filename="check.png", dt = 0.1)

    rays = [Ray([1, 0],
            [0.1, float(i)],
            1.0) for i = 1:ray_num]

    #lenses = [Lens([10, 6], 6, constant)]
    lenses = [Lens([10, 6], 6, constant)]
    #lenses = [Lens([-15, -10], 5, 1.5)]
    mirrors = [Mirror([0.0, -1.0],[5.0, 0.0])]
#=
    mirrors = [Mirror([0.0, 1.0],[5.0, 5.0]),
               Mirror([0.0, -1.0],[5.0, 0.0]),
               Mirror([1.0, 0.0],[0.0, 2.5]),
               Mirror([-1.0, 0.0],[10, 2.5])]
=#

    positions = zeros(ray_num, n, 2)

    propagate!(rays, mirrors, lenses, positions, n, dt)

    plot_rays(positions, mirrors, lenses, filename)

    return rays
end
