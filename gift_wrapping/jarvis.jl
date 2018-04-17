struct Pos
    x::Float64
    y::Float64
end

function jarvis_angle(point1::Pos, point2::Pos, point3::Pos)
    # Find distances between all points
    a = sqrt((point2.x - point3.x)^2 + (point2.y - point3.y)^2)
    b = sqrt((point3.x - point1.x)^2 + (point3.y - point1.y)^2)
    c = sqrt((point1.x - point2.x)^2 + (point1.y - point2.y)^2)

    println(point1)
    println(point2)
    println(point3)
    println(a)
    println(b)
    println(c)
    println(-(c*c - a*a - b*b)/(2*a*b))
    println()
    ret_angle = acos((b*b - a*a - c*c)/(2*a*c))
    println(ret_angle)
    println()

    if(sign(point1.x - point2.x) != sign(point1.x - point3.x))
        ret_angle += 0.5*pi
    end

    if (isnan(ret_angle))
        exit(1)
    end

    return ret_angle
end

function jarvis_march(points::Vector{Pos})
    hull = Vector{Pos}()

    # sorting array based on leftmost point
    sort!(points, by = item -> item.x)
    push!(hull, points[1])

    i = 1
    curr_point = points[2]

    # Find angle between points
    curr_theta = jarvis_angle(Pos(0,0), hull[1], curr_point)
    while (curr_point != hull[1])
        for point in points
            if (hull[i] != point)
                theta = 0.0
                if (i > 1)
                    theta = jarvis_angle(hull[i-1], hull[i], point)
                else
                    theta = jarvis_angle(Pos(0,0), hull[i], point)
                end
    
                if (theta < curr_theta)
                    curr_point = point
                    curr_theta = theta
                end
            end
        end
        push!(hull, curr_point)
        i += 1
    end

    return hull
end

function main()

    points = [Pos(2,1.5), Pos(1, 1), Pos(2, 4), Pos(3, 1)]
    hull = jarvis_march(points)
    println(hull)
end

#=
println("angle is:")
println(jarvis_angle(Pos(0,0), Pos(0,1), Pos(1,1)))
println("angle is:")
println(jarvis_angle(Pos(0,0), Pos(0,1), Pos(2,1)))
=#

main()
