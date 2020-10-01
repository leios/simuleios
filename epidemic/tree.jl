using DataStructures

include("tree_helpers.jl")

mutable struct Node
    neighbors::Vector{Node}
    ID::Int64
    Node(ID::Int64) = new(Vector{Node}(), ID)
end

function DFS_recursive(n::Node)
    # Here we are doing something...
    println(n.ID)

    for neighbor in n.neighbors
        DFS_recursive(neighbor)
    end
end

function DFS_stack(n::Node)
    s = Stack{Node}()
    push!(s, n)

    while(length(s) > 0)
        println(top(s).ID)
        temp = pop!(s)
        for neighbor in temp.neighbors
            push!(s, neighbor)
        end
    end
end

function BFS_queue(n::Node)
    q = Queue{Node}()
    enqueue!(q, n)

    while(length(q) > 0)
        println(front(q).ID)
        temp = dequeue!(q)
        for neighbor in temp.neighbors
            enqueue!(q, neighbor)
        end
    end

end
