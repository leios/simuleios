#-------------genetic.jl-------------------------------------------------------#
#
# Purpose: To implement a simple genetic algorithm for function minimization
#
#------------------------------------------------------------------------------#

using PyPlot

# For now, we will evaluate (x-1)^2
function evaluate(x::Float64)
    return -(x-1)*(x-1)
end

# Initialize num positions centered around origin
function init(num::Int64, box_length::Float64)
    return rand(num) * box_length - box_length * 0.5
end

# Function for actual tournament
function tournament(population::Vector{Float64}, size::Int64)
    # Selecting first individual
    best = rand(population)

    # Choosing best from random selection of others
    for j = 2:size
        candidate = rand(population)
        if (evaluate(candidate) > evaluate(best))
            best = candidate
        end
    end

    return best
end

# Tournament selection, size is tournament size, and prob is prob of selection
# num_parents must be even
# The num_children = num_parents, make sure that 
#     the children size < population size - elite number
function selection(population::Vector{Float64}, size::Int64, 
                   num_parents::Int64)

    if (num_parents % 2 != 0)
        num_parents += 1
    end

    # result is the final selected population
    parents = fill(0.0, num_parents)

    # selection loop
    for i = 1:num_parents
        parents[i] = tournament(population, size)
    end

    return parents
end

# Crossover
function crossover(parents::Vector{Float64}, res::Int64, cross_rate::Float64)

    # vector of offspring to work from
    offspring = fill(0.0, length(parents))
    child1::Float64 = 0.0
    child2::Float64 = 0.0

    # We are selecting two parents and creating two children
    for i = 1:2:length(parents)
        # splitting each parent into segments based on location
        for j = 1:res
           index = rand()
           if (index > cross_rate)
               child1 += parents[i] / res
               child2 += parents[i+1] / res
           else
               child1 += parents[i+1] / res
               child2 += parents[i] / res
           end
        end
        offspring[i] = child1
        offspring[i] = child2
    end

    return offspring
end

# Section of individuals for repopulation
function repopulate(population::Vector{Float64}, parents::Vector{Float64},
                    offspring::Vector{Float64}, elite::Int64, box_size::Float64)

    # grabbing the elite from the previous population
    population = sort(population)

    # Randomizing all elements that are not elite or children
    leftover_num = length(population) - length(offspring) - elite + 1
    if leftover_num < 0
        println("More offspring / elite than in population")
    end
    println(length(parents), '\t', length(offspring), '\t',
            length(population), '\t', elite, '\t', leftover_num)

    leftovers = init(leftover_num, box_size)

    j = 1
    for i = elite:length(population)
        if j <= length(offspring)
            population[i] = offspring[j]
        else
            population[i] = leftovers[j - length(offspring)]
        end
        j += 1
        
    end

    return population
end

# Function for mutation of an individual
function mutation(offspring::Vector{Float64}, mutation_rate::Float64, 
                  box_length::Float64)

    # randomly mutating offspring
    for i = 1:length(offspring)
       if rand() < mutation_rate
           offspring[i] = rand() * box_length - box_length * 0.5
       end 
    end

    return offspring

end

# main function
function main()
    num = 100
    box_length = 5.0
    population::Vector{Float64} = init(num, box_length)
    plot(population)
    parents::Vector{Float64} = selection(population, 10, 6)
    offspring::Vector{Float64} = crossover(parents, 10, .5)
    offspring = mutation(offspring, 0.1, box_length)
    population = repopulate(population, parents, offspring, 10, box_length)
    plot(population)
end

main()
