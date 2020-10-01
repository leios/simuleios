using DataStructures
using LightGraphs

# For plotting
using GraphRecipes
import Plots
Plots.pyplot()

function find_element_num(n::T) where {T}
    s = Stack{T}()
    push!(s, n)

    num = 1
    while(length(s) > 0)
        num += 1
        temp = pop!(s)
        for neighbor in temp.neighbors
            push!(s, neighbor)
        end
    end
    return num
end

# We'll talk about this later if we have time!
function GraphRecipes.graphplot(n::T) where {T}

    # Constructing a LightGraphs graph
    element_num = find_element_num(n)

    g = SimpleGraph(element_num)

    s = Stack{T}()
    push!(s, n)

    while(length(s) > 0)
        temp = pop!(s)

        for neighbor in temp.neighbors
            add_edge!(g,neighbor.ID, temp.ID)
            push!(s, neighbor)
        end
    end
    graphplot(g;curves=false, markersize=0.2, names=1:element_num)

end

function create_tree(num_row::Int64, num_neighbor::Int64; mode="Recursive")

    if mode == "recursive"
        return create_tree(num_row, num_neighbor, [1])
    elseif mode == "stack"
        return create_tree_stack(num_row, num_neighbor, [1])
    elseif mode == "queue"
        return create_tree_queue(num_row, num_neighbor)
    else
        println("error mode ", mode, " not found for tree construction!")
    end
end

function create_tree_queue(num_row::Int64, num_neighbor::Int64)

    # Creating tree
    tree = create_tree(num_row, num_neighbor, [1])

    # relabelling elements
    q = Queue{Node}()
    enqueue!(q, tree)

    ID = 1
    while(length(q) > 0)
        temp = dequeue!(q)
        temp.ID = ID
        ID += 1
        for neighbor in temp.neighbors
            enqueue!(q, neighbor)
        end
    end

    return tree

end

function create_tree_stack(num_row::Int64, num_neighbor::Int64, ID::Vector)
    final_tree = Node(ID[1])
    if (num_row == 0)
        return final_tree
    end

    #preallocating for stack ordering
    for i = 1:num_neighbor
        push!(final_tree.neighbors, Node(ID[1]))
    end

    for i = num_neighbor:-1:1
        ID[1] += 1
        neighbor = create_tree(num_row - 1, num_neighbor, ID)
        final_tree.neighbors[i] = neighbor
    end

    return final_tree
end

function create_tree(num_row::Int64, num_neighbor::Int64, ID::Vector)
    final_tree = Node(ID[1])
    if (num_row == 0)
        return final_tree
    end

    for i = 1:num_neighbor
        ID[1] += 1
        neighbor = create_tree(num_row - 1, num_neighbor, ID)
        push!(final_tree.neighbors, neighbor)
    end

    return final_tree
end
