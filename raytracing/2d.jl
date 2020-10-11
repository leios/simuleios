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

    # Index of Refraction
    ior::Float64
end

function lens_normal_at(ray, lens)
    n = normalize(ray.p .- lens.p)
    if dot(-n, ray.v) < 0
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

# note: light moves as a particular speed with respect to the medium it is
#       moving through, so...
#       n_2*v = n_1*l + (n_1*cos(theta_1) - n_2*cos(theta_2))*n
#       Other approximations: ior = n_1/n_2, c = -n*l
function refract!(ray, lens, ior)
    n = lens_normal_at(ray, lens)
    c = dot(-n, ray.v);
    d = 1.0 - ior^2 * (1.0 - c^2);

    if (d < 0.0)
        return zeros(2);
    end

    ray.v = ior * ray.v + (ior * c - sqrt(d)) * n;
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
    plt = plot(background_color=:black, aspect_ratio=:equal)
    for ray in rays
        plot!(plt, (ray.positions[:,1], ray.positions[:,2]); label = "ray",
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
    ray.p .+= .+ ray.v.*dt
end


function intersection_point(ray::Ray, lens::Lens)
    # distance from the center of the lens
    relative_dist = lens.p - ray.p

    # distance projected onto the direction vector
    projected_relative_dist = dot(relative_dist, ray.v)

    # distance of closest approach
    # Might make more sense to do a dot product instead of a norm
    closest_approach = sqrt(norm(relative_dist)^2 -
                            projected_relative_dist^2)

    if closest_approach >= lens.r
        return nothing
    end

    # half chord length
    half_chord = sqrt(lens.r^2 - closest_approach^2)

    in_lens = half_chord < projected_relative_dist
    if in_lens
        return (projected_relative_dist - half_chord)*ray.v
    else
        return (projected_relative_dist + half_chord)*ray.v
    end
end

function intersect_check()

    ray = Ray([1.0, 0.0], [0.0, 1.0], zeros(2,2))

    lens = Lens([3.0, 1.0], 1.0, 1.5)

    pt = intersection_point(ray, lens)
    println(pt)
end
