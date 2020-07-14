using Plots
using DataStructures

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

function find_neighbors(canvas, loc::CartesianIndex, old_val, new_val;
                       direction = "all")
    
    neighbors =  []
    if direction == "all"
        possible_neighbors = [loc + CartesianIndex(0, 1),
                              loc + CartesianIndex(1, 0),
                              loc + CartesianIndex(0, -1),
                              loc + CartesianIndex(-1, 0)]
    elseif direction == "north"
        temp_loc = loc + CartesianIndex(1, 0)
        possible_neighbors = [temp_loc]
    elseif direction == "south"
        temp_loc = loc + CartesianIndex(-1, 0)
        possible_neighbors = [temp_loc]
    elseif direction == "east"
        temp_loc = loc + CartesianIndex(0, 1)
        possible_neighbors = CartesianIndex[]
        while canvas[temp_loc] == old_val
            push!(possible_neighbors, temp_loc)
            temp_loc = temp_loc + CartesianIndex(0, 1)
        end
    elseif direction == "west"
        temp_loc = loc + CartesianIndex(0, -1)
        possible_neighbors = CartesianIndex[]
        while canvas[temp_loc] == old_val
            push!(possible_neighbors, temp_loc)
            temp_loc = temp_loc + CartesianIndex(0, -1)
        end
    else
        error("direction ", direction, " not found!")
    end
    for possible_neighbor in possible_neighbors
        if inbounds(size(canvas), possible_neighbor) &&
           canvas[possible_neighbor] == old_val
            push!(neighbors, possible_neighbor)
        end
    end

    return neighbors
end

function stack_fill!(canvas, loc::CartesianIndex, old_val, new_val)
    if new_val == old_val
        return
    end

    s = Stack{CartesianIndex}()
    #s = CartesianIndex[]
    push!(s, loc)

    while length(s) > 0
        temp_loc = pop!(s)
        color!(canvas, temp_loc, old_val, new_val)
        possible_neighbors = find_neighbors(canvas, temp_loc, old_val, new_val)
        for neighbor in possible_neighbors
            push!(s,neighbor)
        end
        
    end
end


function queue_fill!(canvas, loc::CartesianIndex, old_val, new_val)
    if new_val == old_val
        return
    end

    q = Queue{CartesianIndex}()
    enqueue!(q, loc)

    while length(q) > 0
        temp_loc = dequeue!(q)
        color!(canvas, temp_loc, old_val, new_val)
        possible_neighbors = find_neighbors(canvas, temp_loc, old_val, new_val)
        for neighbor in possible_neighbors
            enqueue!(q,neighbor)
        end
        
    end
end

function recursive_fill!(canvas, loc::CartesianIndex, old_val, new_val)
    if (old_val == new_val)
        return
    elseif (canvas[loc] != old_val)
        return
    else
        canvas[loc] = new_val
    end

    possible_neighbors = find_neighbors(canvas, loc, old_val, new_val)
    for possible_neighbor in possible_neighbors
        recursive_fill!(canvas, possible_neighbor, old_val, new_val)
    end
end

# TODO: find array of *all* west and east elements in find_neighbor
function loop_fill!(canvas, loc::CartesianIndex, old_val, new_val)

    if (old_val == new_val)
        return
    elseif (canvas[loc] != old_val)
        return
    end

    q = Queue{CartesianIndex}()
    enqueue!(q, loc)

    for element in q
        west_elements = find_neighbors(canvas, element, old_val, new_val;
                       direction = "west")
        east_elements = find_neighbors(canvas, element, old_val, new_val;
                       direction = "east")
        for temp_loc in vcat(west_elements, east_elements)
            color!(canvas, temp_loc, old_val, new_val)

            north_element = find_neighbors(canvas, temp_loc, old_val,
                            new_val; direction = "north")
            south_element = find_neighbors(canvas, temp_loc, old_val,
                            new_val; direction = "south")

            if length(north_element) > 0
                enqueue!(q,north_element[1])
            end
            if length(south_element) > 0
                enqueue!(q,south_element[1])
            end
        end
        
        color!(canvas, element, old_val, new_val)
    end
end

function init_canvas()
    canvas = zeros(100,100)
    loc = CartesianIndex(50,50)
    create_circle!(canvas, loc, 7, 2, 0.75)
    return canvas
end

function main()
    canvas = init_canvas()
    loc2 = CartesianIndex(50,50)
    #recursive_fill!(canvas, loc2, 0.0, 0.5)
    #queue_fill!(canvas, loc2, 0.0, 0.5)
    #stack_fill!(canvas, loc2, 0.0, 0.5)
    loop_fill!(canvas, loc2, 0.0, 0.5)

    heatmap(canvas, color=:coolwarm)
end

main()
