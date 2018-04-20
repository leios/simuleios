struct Pos
    x::Float64
    y::Float64
end

function jarvis_angle(point1::Pos, point2::Pos, point3::Pos)
    vec1 = Pos(point2.x - point1.x, point2.y - point1.y)
    vec2 = Pos(point3.x - point2.x, point3.y - point2.y)
    mag1 = sqrt(vec1.x*vec1.x + vec1.y*vec1.y)
    mag2 = sqrt(vec2.x*vec2.x + vec2.y*vec2.y)
    ret_angle = acos((vec1.x*vec2.x + vec1.y*vec2.y)/(mag1*mag2))
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
    #while (curr_point != hull[1])
    while (length(hull) < 4)
        println(hull)
        for point in points
                theta = 0.0
            if (i == 1)
                if (hull[i] != point)
                    theta = jarvis_angle(Pos(0,0), hull[i], point)
                end
            else
                if (hull[i] != point && hull[i-1] != point)
                    theta = jarvis_angle(hull[i-1], hull[i], point)
                end
            end
            println(hull[i])
            println(point)
            println(curr_theta)
            println(theta)
            println()
            if (theta > curr_theta)
                curr_point = point
                curr_theta = theta
            end
        end
        push!(hull, curr_point)
        curr_theta = 0
        i += 1
    end

    return hull
end

function main()

    points = [Pos(2,1.5), Pos(1, 1), Pos(2, 4), Pos(3, 1)]
    hull = jarvis_march(points)
    println(hull)
end

println("angle is:")
println(jarvis_angle(Pos(0,0), Pos(1,0), Pos(1,1)))
println("angle is:")
println(jarvis_angle(Pos(0,0), Pos(1,0), Pos(2,1)))

main()
