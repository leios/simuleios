# note: for reflection, l_x -> l_x, but l_y -> -l_y
#       In this case, the y component of l = cos(theta)*n
#       so new vector: v = l + 2cos(theta)*n

# note: light moves as a particular speed with respect to the medium it is
#       moving through, so...
#       n_2*v = n_1*l + (n_1*cos(theta_1) - n_2*cos(theta_2))*n
#       Other approximations: r = n_1/n_2, c = -n*l

using Plots

mutable struct Ray
    v::Tuple{Float64, Float64}
    p::Tuple{Float64, Float64}
    positions::Array{Tuple{Float64, Float64}}
end

function plot_rays(rays::Vector{Ray}, filename)
    plt = plot()
    for ray in rays
        plot!(plt, ray.positions)
    end

    savefig(plt, filename)
end

function step!(ray::Ray, dt)
    # TODO: implement refraction / reflection
    ray.v = ray.v
    ray.p = ray.p .+ ray.v.*dt
end

function propagate!(rays::Vector{Ray}, n, dt)

    for i = 1:n
        for j = 1:length(rays)
            step!(rays[j], dt)
            rays[j].positions[i] = rays[j].p
        end
    end
end

function parallel_propagate(ray_num, n)

    rays = [Ray((0.0, 1.0),
            (float(i), 0.0),
            Array{Tuple{Float64, Float64}}(undef, n)) for i = 1:ray_num]

    propagate!(rays, n, 1.0)

    return rays
end
