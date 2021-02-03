using Makie, GLMakie, ModernGL
function rotate(p::Array{N}, angle) where {N <: Number}

    matrix = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1]

    output = matrix*vcat(p, 1)
    return output[1:2]
end

function plot_makie()
    points = [Point{3,Float64}(vcat(rotate([1,0],2*pi*i/100), 1)) for i = 1:101]

    for i = 1:100
        scene = Scene()
        angle = 2*pi*i/100
        point = Point{3,Float64}(vcat(rotate([1,0], angle), 1))
        scatter!(scene, point, color = :red, markersize = 50)
        lines!(scene, points, color = :black)
        cam3d!(scene)
        zlims!(scene, (0,2))

        save("scene"*lpad(i, 4, "0")*".png", scene; resolution = (600,400))
    end
end
