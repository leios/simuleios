# TODO: abstract to objects (spheres) with Transmission and Reflectivity

using Plots
using LinearAlgebra
using Test

pyplot()

function Base.isapprox(n1::Nothing, n2::Nothing)
    return true
end

abstract type Object end

mutable struct Ray
    # Velocity vector
    v::Vector{Float64}

    # Position vector
    p::Vector{Float64}

    # Set of positions
    positions::Array{Float64}

end

mutable struct Lens <: Object
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

abstract type Wall <: Object end;

mutable struct Mirror <: Wall
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
              linecolor=:yellow)
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

function intersection(ray::Ray, lens::Lens)
    # distance from the center of the lens
    relative_dist = lens.p - ray.p

    # distance projected onto the direction vector
    projected_relative_dist = dot(relative_dist, ray.v)

    # distance of closest approach
    # Might make more sense to do a dot product instead of a norm
    closest_approach = abs(sqrt(norm(relative_dist)^2 -
                            projected_relative_dist^2))

    # TODO: arbitrary threshold, should be fixed.
    if abs(closest_approach - lens.r) < 0.001
        closest_approach = lens.r
    end 

    if closest_approach > lens.r
        return nothing
    end

    # half chord length
    half_chord = sqrt(lens.r^2 - closest_approach^2)

    # checks if ray is moving away from sphere by seeing if we are 
    # outside of sphere (abs(projected_relative_dist) > half_chord), and if
    # the ray is moving away from the boundary
    if (projected_relative_dist < 0 &&
        abs(projected_relative_dist) > half_chord)
        return nothing
    end 

    in_lens = half_chord < projected_relative_dist
    intersect = zeros(2)
    if in_lens
        intersect = (projected_relative_dist - half_chord)*ray.v
    else
        intersect = (projected_relative_dist + half_chord)*ray.v
    end

    return intersect
end

function intersection_quadratic(ray::Ray, lens::Lens)
    relative_dist = ray.p-lens.p
    a = dot(ray.v, ray.v)
    b = 2.0 * dot(relative_dist, ray.v)
    c = dot(relative_dist, relative_dist) - lens.r*lens.r
    discriminant = b*b - 4*a*c

    if discriminant < 0
        return nothing
    elseif discriminant > 0
        roots = [(-b + sqrt(discriminant)) / (2*a),
                 (-b - sqrt(discriminant)) / (2*a)]
        min = minimum(roots)
        max = maximum(roots)

        if min >= 0
            return (min)*ray.v
        elseif max >= 0
            return (max)*ray.v
        else
            return nothing
        end
    else
        return (-b/(2*a))*ray.v
    end 
end

function intersection(ray::Ray, wall::W) where {W <: Wall}
    intersection = -dot((ray.p .- wall.p),wall.n)/dot(ray.v, wall.n)

    if isfinite(intersection) && intersection > 0 && intersection != NaN
        return intersection*ray.v
    else
        return nothing
    end
end

function intersect_test()

    lens = Lens([3.0, 1.0], 1.0, 1.5)

    @testset "intersect lens tests" begin
        rays = [Ray([1.0, 0.0], [0.0, 1.0], zeros(2,2)),
                Ray([1.0, 0.0], [3.0, 1.0], zeros(2,2)),
                Ray([sqrt(0.5), sqrt(0.5)], [2.0, 0.0], zeros(2,2)),
                Ray([-1.0, 0.0], [0.0, 1.0], zeros(2,2)),
                Ray([0.0, 1.0], [0.0, 1.0], zeros(2,2)),
                Ray([1.0, 0.0], [0.0, 0.0], zeros(2,2))]
        answers = [[2.0, 0.0], [1.0, 0.0], [1-sqrt(0.5), 1-sqrt(0.5)],
                   nothing, nothing, [3.0, 0.0]]

        for i = 1:length(rays)
            pt = intersection(rays[i], lens)
            @test isapprox(pt, answers[i])
            pt = intersection_quadratic(rays[i], lens)
            @test isapprox(pt, answers[i])
        end
    end

    wall = Mirror([-1.0, 0.0], [2.0, 1.0])

    @testset "intersect wall tests" begin
        rays = [Ray([1.0, 0.0], [0.0, 1.0], zeros(2,2)),
                Ray([sqrt(0.5), sqrt(0.5)], [0.0, 0.0], zeros(2,2)),
                Ray([-sqrt(0.5), sqrt(0.5)], [0.0, 0.0], zeros(2,2)),
                Ray([0.0, 1.0], [0.0, 1.0], zeros(2,2))]

        answers = [[2.0, 0.0], [2.0, 2.0], nothing, nothing]

        for i = 1:length(rays)
            pt = intersection(rays[i], wall)
            @test isapprox(pt, answers[i])
        end
    end
end

function propagate!(rays::Vector{Ray}, objects::Vector{O},
                    num_intersections) where {O <: Object}
    for i = 2:num_intersections
        for j = 1:length(rays)

            intersect_final = [Inf, Inf]
            intersected_object = nothing
            for object in objects
                intersect = intersection(rays[j], object)
                if intersect != nothing &&
                   sum(intersect[:].^2) < sum(intersect_final[:].^2)
                    intersect_final = intersect
                    intersected_object = object
                end
            end

            if intersect_final != nothing
                rays[j].p .+= intersect_final
                rays[j].positions[i,:] .= rays[j].p
                if typeof(intersected_object) == Lens
                    ior = intersected_object.ior
                    if inside_of(rays[j], intersected_object)
                        ior = 1/intersected_object.ior
                    end

                    refract!(rays[j], intersected_object, ior)
                elseif typeof(intersected_object) == Mirror
                    reflect!(rays[j], intersected_object.n)
                end

                println(rays[j].v, '\t', rays[j].p)
            end
        end
        println('\n')
    end
end

function parallel_propagate(ray_num, num_intersections, box_size;
                            filename="check.png")

    rays = [Ray([1, 0],
            [0.1, float(i)],
            zeros(num_intersections, 2)) for i = 1:ray_num]

    # TODO: do this in constructor
    for i = 1:length(rays)
        rays[i].positions[1,:] .= rays[i].p
    end

    lenses = [Lens([10, 5], 5, 1.5)]
    mirrors = [Mirror([-1.0, 0.0], [25.0, 5.0])]
               #Mirror([1.0, 0.0], [0, 5.0]),
               #Mirror([0.0, 1.0], [12.5, 0.0]),
               #Mirror([0.0, -1.0], [12.5, 10.0])]

    objects = vcat(lenses, mirrors)

    propagate!(rays, objects, num_intersections)

    plot_rays(rays, mirrors, lenses, filename)

end
