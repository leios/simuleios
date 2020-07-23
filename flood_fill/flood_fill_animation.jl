using Plots
using ImageMagick
using FileIO
using Images

using DataStructures

import Plots: default
default(show=false, reuse=true)

ENV["GKSwstype"]="nul"

scale = 2
fnt = font("Computer Modern", 10*scale)
default(titlefont=fnt, guidefont=fnt, tickfont=fnt, legendfont=fnt)

function convert_to_float(a::RGB)
    a = float(a)
    return (a.r + a.g + a.b)/3.0
end

function maze_to_canvas(filename)
    img = load(filename)
    return convert_to_float.(img)
end

function inbounds(canvas_size, loc)
    if minimum(Tuple(loc)) < 1
        return false
    elseif loc[2] > canvas_size[2]
        return false
    elseif loc[1] > canvas_size[1]
        return false
    else
        return true
    end
end

function create_circle!(canvas, loc::CartesianIndex, radius, threshold,
                        old_val)
    for i = 1:size(canvas)[1]
        for j = 1:size(canvas)[2]
            if ((radius-threshold)^2
                < (i-loc[1])^2+(j-loc[2])^2
                < (radius+threshold)^2)
                canvas[i,j] = old_val
            end
        end
    end
end

function color!(canvas, loc::CartesianIndex, old_val, new_val)
    # bounds check
    if !inbounds(size(canvas), loc)
        return
    end

    if (old_val == new_val)
        return
    elseif (canvas[loc] != old_val)
        return
    else
        canvas[loc] = new_val
    end
end

function find_neighbors(canvas, loc::CartesianIndex, old_val, new_val)
    
    neighbors =  []
    possible_neighbors = [loc + CartesianIndex(0, 1),
                          loc + CartesianIndex(1, 0),
                          loc + CartesianIndex(0, -1),
                          loc + CartesianIndex(-1, 0)]
    for possible_neighbor in possible_neighbors
        if inbounds(size(canvas), possible_neighbor) &&
           canvas[possible_neighbor] == old_val
            push!(neighbors, possible_neighbor)
        end
    end

    return neighbors
end

function stack_fill!(canvas, loc::CartesianIndex, old_val, new_val;
                     output=false)
    if new_val == old_val
        return
    end

    s = Stack{CartesianIndex}()
    #s = CartesianIndex[]
    push!(s, loc)

    id = 0

    while length(s) > 0
        temp_loc = pop!(s)
        if canvas[temp_loc] == old_val
            color!(canvas, temp_loc, old_val, new_val)
            if output
                plt = heatmap(canvas, color=:coolwarm, ratio=:equal,
                              size=(1000,1000))
                savefig("out"*lpad(string(id[1]),5,string(0))*".png") 
                id += 1
                println("Outputting: ", id)
            end 
            possible_neighbors = find_neighbors(canvas, temp_loc,
                                                old_val, new_val)
            for neighbor in possible_neighbors
                push!(s,neighbor)
            end
        end
        
    end
end


function queue_fill!(canvas, loc::CartesianIndex, old_val, new_val;
                     output=false)
    if new_val == old_val
        return
    end

    q = Queue{CartesianIndex}()
    enqueue!(q, loc)
    color!(canvas, loc, old_val, new_val)

    id = 0

    while length(q) > 0
        temp_loc = dequeue!(q)
        if output
            plt = heatmap(canvas, color=:coolwarm, ratio=:equal,
                          size=(500,500))
            savefig("out"*lpad(string(id[1]),5,string(0))*".png") 
            id += 1
            println("Outputting: ", id)
        end 

        possible_neighbors = find_neighbors(canvas, temp_loc, old_val, new_val)
        for neighbor in possible_neighbors
            color!(canvas, neighbor, old_val, new_val)
            enqueue!(q,neighbor)
        end
        
    end
end

function maze_fill!(canvas, loc::CartesianIndex, old_val, new_val;
                     output=false, end_loc = CartesianIndex(-1,-1))
    if new_val == old_val
        return
    end

    q = Queue{CartesianIndex}()
    enqueue!(q, loc)
    color!(canvas, loc, old_val, new_val)

    id = 0

    stop_solve = false
    while !stop_solve
        temp_loc = dequeue!(q)
        if output
            plt = heatmap(canvas, ratio=:equal,
                          size=(500,500), legend=:bottomright)
            scatter!(plt, Tuple(loc), label="Start")
            scatter!(plt, Tuple(end_loc), label="Finish")
            savefig("out"*lpad(string(id[1]),5,string(0))*".png")
            id += 1
            println("Outputting: ", id)
        end 

        possible_neighbors = find_neighbors(canvas, temp_loc, old_val, new_val)
        for neighbor in possible_neighbors
            color!(canvas, neighbor, old_val, new_val)
            enqueue!(q,neighbor)
        end

        if (length(q) == 0 || temp_loc == end_loc)
            stop_solve = true
        end
        
    end
end


function recursive_fill!(canvas, loc::CartesianIndex, old_val, new_val;
                         output=false, id=id)

    if (old_val == new_val)
        return
    elseif (canvas[loc] != old_val)
        return
    else
        canvas[loc] = new_val
    end

    if output
        plt = heatmap(canvas, color=:coolwarm, ratio=:equal, size=(1000,1000))
        savefig("out"*lpad(string(id[1]),5,string(0))*".png") 
        id[1] += 1
        println("Outputting: ", id[1])
    end 

    possible_neighbors = find_neighbors(canvas, loc, old_val, new_val)
    for possible_neighbor in possible_neighbors
        recursive_fill!(canvas, possible_neighbor, old_val, new_val;
                        output=output,id=id)
    end
end

function init_canvas()
    canvas = zeros(30,30)
    loc = CartesianIndex(15,15)
    create_circle!(canvas, loc, 12, 2, 0.75)
    return canvas
end

function create_grid(dims)
    tuple_array = Array{Tuple{Int, Int}}(undef, dims[1]*dims[2])
    grid = Array{Float64}(undef, dims)
    for i = 1:dims[1]
        for j = 1:dims[2]
            tuple_array[(i-1)*dims[2]+j] = (i,j)
            grid[i,j] = i+j
        end
    end

    lines = Array{Tuple{Int, Int}}(undef, 10,2)
    for i = 1:5
        lines[i,:]=[tuple_array[(i-1)*5+1], tuple_array[i*5]]
        lines[i+5,:]=[tuple_array[i], tuple_array[20+i]]
    end 

    println(lines)

    plt = heatmap(grid, color=:coolwarm, size=(600,500), aspect_ratio=:equal,
                 xlims=(0.5,5.5), ylims=(0.5,5.5))
    for i = 1:size(lines)[1]
        plot!(plt, lines[i,:], label="", color=:black, linewidth=3)
    end
    #plot!(plt, lines, label="")
    scatter!(plt, tuple_array, label = "", markersize=15)

    savefig("grid.png")

end

function main()

    create_grid((5,5))
#=
    canvas = init_canvas()
    loc2 = CartesianIndex(15,15)
    #recursive_fill!(canvas, loc2, 0.0, 0.5; output=true, id = [0])
    #queue_fill!(canvas, loc2, 0.0, 0.5; output=true)
    #stack_fill!(canvas, loc2, 0.0, 0.5; output=true)

    plt = heatmap(canvas, color=:coolwarm, size=(600,500), aspect_ratio=:equal,
                  xlims=(1,30), ylims=(1,30))
    savefig("simple_circle.png")

    # For maze
    maze = maze_to_canvas("maze2.png")
    maze_start = CartesianIndex(50,50)
    maze_end = CartesianIndex(96,96)
    maze_fill!(maze, maze_start, 1.0, 0.5; end_loc = maze_end, output=true)
    heatmap(maze)
=#

end

main()
