@enum Int64 S I R

include("tree_helpers.jl")

# Note: I usually define a helper function to more easily add edges
mutable struct Person
    SIR
    ID
    is_infected::Bool
    neighbors::Vector{Person}
    Person(SIR, ID) = new(SIR, ID, false, [])
end

function infect(p::Person)
    if p.SIR == S
        p.is_infected = true
        p.SIR = I
        for neighbor in p.neighbors
            infect(neighbor)
        end
    else
        return
    end

end

function sir_test()
    people = [Person(S,i) for i = 1:6]

    # setting the 4th person to be recovered
    people[4].SIR = R

    # this is where the helper function would allow us to specify only 1 dir!
    people[1].neighbors = [people[2], people[3]]
    people[2].neighbors = [people[3]]
    people[3].neighbors = [people[4], people[5]]
    people[5].neighbors = [people[6]]

    infect(people[1])

    for person in people
        println("infection status: ", person.is_infected)
    end

    return people
end
