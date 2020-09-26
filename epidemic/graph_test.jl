using LightGraphs

mutable struct Node
    push::Bool
    pull::Bool
    weight::Float64
    neighbors::Vector{Node}
end

function infect(n::Node, new_weight)
    if n.pull == true
        n.weight = new_weight
        n.pull = false
    end

    if n.push == true
        n.push = false
        for neighbor in n.neighbors
            infect(neighbor, n.weight)
        end
    end

end

#TODO: write function to more easily add neighbors

function test()
    a = Node(true, true, 0.0, [])
    b = Node(false, true, 0.0, [a])
    c = Node(true, true, 0.0, [])
    d = Node(false, false, 0.0, [c])
    e = Node(true, true, 0.0, [])
    f = Node(false, true, 0.0, [e])

    a.neighbors = [b, c]
    c.neighbors = [e]
    e.neighbors = [f]

    infect(a, 1.0)

    println("a weight is: ", a.weight)
    println("b weight is: ", b.weight)
    println("c weight is: ", c.weight)
    println("d weight is: ", d.weight)
    println("e weight is: ", e.weight)
    println("f weight is: ", f.weight)
end
