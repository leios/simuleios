function random_dist(n::Int64)
    point = [0.0, 0.0]
    f = open("random.dat", "w")

    for i = 1:n
        point = [rand(),rand()]
        println(f, point[1], '\t', point[2],"\n\n")
    end

    close(f)
end

function barnsley_tree(n::Int64)
    point = [0.0, 0.0]
    f = open("barnsley_tree.dat", "w")
    for i = 1:n
        rnd = rand()
        if (rnd <= 0.02)
            point = [0.03 0; 0 0.1]*point
        elseif(rnd > 0.02 && rnd <= 0.62)
            point = [0.85 0.0; 0.0 0.85]*point + [0, 1.50]
        elseif(rnd > 0.62 && rnd <= 0.72)
            point = [0.8 0; 0 0.8]*point + [0, 1.50]
        elseif(rnd > 0.72 && rnd <= 0.79)
            point = [0.2 -0.08; 0.15 0.22]*point + [0, 0.85]
        elseif(rnd > 0.79 && rnd <= 0.86)
            point = [-0.2 0.08; 0.15 0.22]*point + [0, 0.85]
        elseif(rnd > 0.86 && rnd <= 0.93)
            point = [0.25 -0.1; 0.12 0.25]*point + [0, 0.3]
        else
            point = [-0.2 0.1; 0.12 0.2]*point + [0, 0.4]
        end

        println(f, point[1], '\t', point[2])

        if (i%10000==0)
            println(f,"\n\n")
        end
    end
    close(f)
end

function barnsley_fern(n::Int64)
    point = [0.0, 0.0]
    f = open("barnsley.dat", "w")
    for i = 1:n
        rnd = rand()
        if (rnd <= 0.01)
            point = [0 0; 0 0.16]*point
        elseif(rnd > 0.01 && rnd <= 0.86)
            point = [0.85 0.04; -0.04 0.85]*point + [0, 1.60]
        elseif(rnd > 0.86 && rnd <= 0.93)
            point = [0.2 -0.26; 0.23 0.22]*point + [0, 1.60]
        else
            point = [-0.15 0.28; 0.26 0.24]*point + [0, 0.44]
        end

        println(f, point[1], '\t', point[2])

        if (i%10000==0)
            println(f,"\n\n")
        end
    end
    close(f)
end

function sierpensky(n::Int64)
    element_x = rand()
    element_y = rand()
    point = [element_x element_y]

    f = open("sierpensky.dat", "w")

    for i = 1:n
        if (i > 20)
            println(f, point[1], '\t', point[2])
        end
        rnd = rand()

        if (rnd <= 0.33)
            point = 0.5*point
        elseif(rnd > 0.33 && rnd <= 0.66)
            point[1] = (point[1]+0.5)/2
            point[2] = (point[2]+1)/2
        else
            point[1] = (point[1]+1)/2
            point[2] = (point[2])/2
        end
    end

    close(f)
end

#random_dist(1000)
#barnsley_fern(1000000)
barnsley_tree(1000000)
#sierpensky(10000)
