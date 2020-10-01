using LightGraphs

include("tree_helpers.jl")

@enum Int64 S I R

mutable struct Person
    SIR
    is_infected::Bool
    Person(SIR) = new(SIR, false)
end

function infect(people, graph, infection_pt)
    if people[infection_pt].SIR == S
        people[infection_pt].is_infected = true
        people[infection_pt].SIR = I
        for neighbor in graph.fadjlist[infection_pt]
            infect(people, graph, neighbor)
        end
    else
        return
    end

end

function sir_test()
    people = [Person(S) for i = 1:6]

    # setting the 4th person to be recovered
    people[4].SIR = R

    # creating a graph to work with
    g = SimpleGraph(6)

    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 4)
    add_edge!(g, 3, 5)
    add_edge!(g, 5, 6)

    infect(people, g, 1)

    for person in people
        println("infection status: ", person.is_infected)
    end

    return g
end
