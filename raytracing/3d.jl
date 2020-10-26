# TODO: get refraction to work without offsets in quadratic / inside_of fxs
#       Also, use dot product to see if exiting lens

using Plots
using LinearAlgebra
using Test

pyplot()

using Images

function Base.isapprox(n1::Nothing, n2::Nothing)
    return true
end

# For now, all cameras are aligned on the z axis
struct Camera
    # Set of all pixels, counts as scene resolution
    pixels

    # physical size of aperture
    size::Vector{Float64}

    # camera's distance from screen
    focal_length::Float64

    # camera's position
    p::Vector{Float64}
end

struct Ray
    # Velocity vector
    v::Vector{Float64}

    # Position vector
    p::Vector{Float64}

    # Color
    c::RGB

end

struct Surface

    # Reflectivity
    r::Float64

    # Transmission
    t::Float64

    # Color
    c::RGBA

    # index of refraction
    ior::Float64

    function Surface(in_r, in_t, in_c, in_ior)
        if !isapprox(in_r+in_t+in_c.alpha, 1)
            error("invalid surface definition, RTC < 1")
        end
        new(in_r,in_t,in_c, in_ior)
    end

    Surface(in_r, in_t, in_c::Float64, in_ior) =
         new(in_r, in_t, RGBA(0,0,0,0), in_ior)
end

abstract type Object end

struct Sphere <: Object
    # Lens position
    p::Vector{Float64}

    # Lens radius
    r::Float64

    s::Surface
end

function Lens(p, r, ior)
    return Sphere(p, r, Surface(0,1,RGBA(0,0,0,0),ior))
end

function ReflectingSphere(p, r)
    return Sphere(p,r,Surface(1,0,RGBA(0,0,0,0),0))
end

function ColoredSphere(p, r, c::RGB)
    return Sphere(p, r, Surface(0,0,RGBA(c), 0))
end

mutable struct SkyBox <: Object
    # Skybox position
    p::Vector{Float64}

    # Skybox radius
    r::Float64
end

function sphere_normal_at(ray, sphere)
    n = normalize(ray.p .- sphere.p)

    return n
end

function inside_of(ray::Ray, sphere)
    return inside_of(ray.p, sphere)
end

function inside_of(pos, sphere)
    x = sphere.p[1] - pos[1]
    y = sphere.p[2] - pos[2]
    if (x^2 + y^2 <= sphere.r^2)
        return true
    else
        return false
    end
end

# note: light moves as a particular speed with respect to the medium it is
#       moving through, so...
#       n_2*v = n_1*l + (n_1*cos(theta_1) - n_2*cos(theta_2))*n
#       Other approximations: ior = n_1/n_2, c = -n*l
function refract(ray, lens::Sphere, ior)
    n = sphere_normal_at(ray, lens)

    if dot(n, ray.v) > 0
        n .*= -1
    end
    c = dot(-n, ray.v);
    d = 1.0 - ior^2 * (1.0 - c^2);

    if (d < 0.0)
        reflect!(ray, n)
        return
    end

    ray_vel = ior * ray.v + (ior * c - sqrt(d)) * n;
    return Ray(ray_vel, ray.p, ray.c)
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
function reflect(ray, n)
    ray_vel = ray.v .- 2*dot(ray.v, n).*n
    ray_pos = ray.p .+ 0.001*ray.v
    return Ray(ray_vel, ray_pos, ray.c)
end

function draw_circle(p, r, res)
    return [x .+ (p[1], p[2]) for x in Plots.partialcircle(0, 2pi, res, r)]
end

function plot_rays(positions, objects::Vector{O},
                   filename) where {O <: Object}
    plt = plot(background_color=:black, aspect_ratio=:equal, legend=false)

    for i = 1:size(positions)[1]
        plot!(plt, (positions[i,:,1], positions[i,:,2]); label = "ray",
              linecolor=:white)
    end

    for object in objects
        if typeof(object) == Mirror
            dir = [-object.n[2], object.n[1]]
            extents = zeros(2,2)
            extents[1,:] .= object.p .- object.scale * dir
            extents[2,:] .= object.p .+ object.scale * dir
            plot!(plt, (extents[:,1], extents[:,2]), label = "mirror")
        elseif typeof(object) == Sphere
            circle = draw_circle(object.p, object.r, 100)
            plot!(circle; label="lens", linecolor=:lightblue)
        end 
    end

    savefig(plt, filename)
end

function step(ray::Ray, dt)
    ray.p .+= .+ ray.v.*dt
    return ray
end

function intersection(ray::Ray, sphere::S;
                      threshold = 0.01) where
                      {S <: Union{Sphere, SkyBox}}
    relative_dist = ray.p-sphere.p
    a = dot(ray.v, ray.v)
    b = 2.0 * dot(relative_dist, ray.v)
    c = dot(relative_dist, relative_dist) - sphere.r*sphere.r
    discriminant = b*b - 4*a*c

    if discriminant < 0
        return nothing
    elseif discriminant > 0
        roots = [(-b + sqrt(discriminant)) / (2*a),
                 (-b - sqrt(discriminant)) / (2*a)]
        min = minimum(roots)
        max = maximum(roots)

        if min > threshold
            return (min)*ray.v
        elseif max > threshold
            return (max)*ray.v
        else
            return nothing
        end
    else
        # Returns nothing if tangential
        return nothing
        #return (-b/(2*a))*ray.v
    end 
end


function intersection(ray::Ray, wall::W) where {W <: Wall}
    intersection_pt = -dot((ray.p .- wall.p),wall.n)/dot(ray.v, wall.n)

    if isfinite(intersection_pt) && intersection_pt > 0 &&
       intersection_pt != NaN
        return intersection_pt*ray.v
    else
        return nothing
    end
end

function propagate(rays::Array{Ray}, objects::Vector{O},
                    num_intersections) where {O <: Object}
    for j = 1:length(rays)
        rays[j] = propagate(rays[j], objects, num_intersections)
    end

    return rays
end

function propagate(ray::Ray, objects::Vector{O},
                   num_intersections) where {O <: Object}

    for i = 1:num_intersections
        if ray.v != zeros(length(ray.v))
            intersect_final = [Inf, Inf]
            intersected_object = nothing
            for object in objects
                intersect = intersection(ray, object)
                if intersect != nothing &&
                   sum(intersect[:].^2) < sum(intersect_final[:].^2)
                    intersect_final = intersect
                    intersected_object = object
                end
            end

            if intersect_final != nothing
                ray.p .+= intersect_final
                if typeof(intersected_object) == Sphere
                    if !isapprox(intersected_object.s.t, 0)
                        ior = 1/intersected_object.s.ior
                        if dot(ray.v,
                               sphere_normal_at(ray,
                                                intersected_object)) > 0
                            ior = intersected_object.s.ior
                        end

                        ray = refract(ray, intersected_object, ior)
                    elseif !isapprox(intersected_object.s.r, 0)
                        n = sphere_normal_at(ray, intersected_object)
                        ray = reflect(ray, n)
                    elseif !isapprox(intersected_object.s.c.alpha, 0)
                        ray_color = RGB(intersected_object.s.c)
                        ray_vel = zeros(length(ray.v))
                        ray = Ray(ray_vel, ray.p, ray_color)
                    end

                elseif typeof(intersected_object) == Mirror
                    ray = reflect(ray, intersected_object.n)
                elseif typeof(intersected_object) == SkyBox
                    ray_color = pixel_color(ray.p)
                    ray_vel = zeros(length(ray.v))
                    ray = Ray(ray_vel, ray.p, ray_color)
                end
            else
                println("hit nothing")
            end
        end
    end

    return ray
end

function pixel_color(position)
    extents = 1000.0
    c = RGB(0)
    if position[1] < extents && position[1] > -extents
        c += RGB((position[1]+extents)/(2.0*extents), 0, 0)
    else
        println(position)
    end

    if position[2] < extents && position[2] > -extents
        c += RGB(0,0,(position[2]+extents)/(2.0*extents))
    else
        println(position)
    end

    if position[3] < extents && position[3] > -extents
        c += RGB(0,(position[3]+extents)/(2.0*extents), 0)
    else
        println(position)
    end

    return c
end

function convert_to_img(rays::Array{Ray}, filename)
    color_array = Array{RGB}(undef, size(rays)[2], size(rays)[1])
    for i = 1:length(color_array)
         color_array[i] = rays[i].c
    end

    save(filename, color_array)
end

function init_rays(cam::Camera)

    res = size(cam.pixels)
    dim = cam.size

    pixel_width = dim ./ res

    # create a set of rays that go through every pixel in our grid.
    rays = Array{Ray}(undef, res[1], res[2])
    for i = 1:res[1]
        for j = 1:res[2]
            pixel_loc = [cam.p[1] + 0.5*dim[1] - i*dim[1]/res[1] + 
                         0.5*pixel_width[1],
                         cam.p[2] + 0.5*dim[2] - j*dim[2]/res[2] +
                         0.5*pixel_width[2],
                         cam.p[3]+cam.focal_length]
            l = normalize(pixel_loc - cam.p)
            rays[res[2]*(i-1) + j] = Ray(l, pixel_loc, RGB(0))
        end
    end

    return rays

end

function ray_trace(objects::Vector{O}, cam::Camera; filename="check.png",
                   num_intersections = 10) where {O <: Object}

    rays = init_rays(cam)

    rays = propagate(rays, objects, num_intersections)

    convert_to_img(rays, filename)

    return rays

end

function main()
    sky = [SkyBox([0.0, 0.0, 0.0], 1000)]
    spheres = [Lens([50,0,-25], 20, 1.5), ReflectingSphere([0,0,-25],20),
               ColoredSphere([-50,0,-25], 20, RGB(0.25, 1, 0.75))]

    objects = vcat(sky, spheres)

    blank_img = Array{RGB}(undef, 1920,1080)
    #blank_img = Array{RGB}(undef, 20, 10)
    blank_img[:] .= RGB(0)

    cam = Camera(blank_img, [160,90], -100, [0,0,100])

    return ray_trace(objects, cam)

end
