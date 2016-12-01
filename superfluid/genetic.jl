#-------------genetic.jl-------------------------------------------------------#
#
# Purpose: To implement a simple genetic algorithm for vortex_location
#
#------------------------------------------------------------------------------#

using PyPlot
include("vortex.jl")

type Circle
    x::Float64
    y::Float64
    radius::Float64
end

function index_check(index::Int64, max_size::Int64)
    if (index < 1)
        index = 1
    end

    if (index > max_size)
        index = max_size
    end
    return index
end

# Fitness of each individual circle
function fitness(cir::Circle, vel::Field)
    # Assume the circle is only 4 points (l1 norm)
    # find integer location of our circle in Field coordinates
    x = round(cir.x / vel.dx)
    y = round(cir.y / vel.dy)

    sum = Position(0,0,0)

    # Extending our circle to the right
    arc = Position(round((cir.x+cir.radius)/vel.dx),y,0)
    index = Int64(arc.y + arc.x * vel.yDim)
    index = index_check(index, length(vel.x))
    #println(arc.y, '\t', arc.x, '\t', vel.yDim)
    #println("circle x and rad are: ", cir.x, '\t', cir.radius, '\t', index)
    sum.x += vel.x[index]
    sum.y += vel.y[index]
    #println(vel.x[index], '\t', vel.y[index])

    # Extending upwards
    arc = Position(x, round((cir.y+cir.radius)/vel.dy),0)
    index = Int64(arc.y + arc.x * vel.yDim)
    index = index_check(index, length(vel.x))
    #println("circle y and rad are: ", cir.y, '\t', cir.radius, '\t', index)
    sum.x += vel.x[index]
    sum.y += vel.y[index]
    #println(vel.x[index], '\t', vel.y[index])
    
    # Extending our circle to the right
    arc = Position(round((cir.x-cir.radius)/vel.dx),y,0)
    index = Int64(arc.y + arc.x * vel.yDim)
    index = index_check(index, length(vel.x))
    #println("circle x and rad are: ", cir.x, '\t', cir.radius, '\t', index)
    sum.x += vel.x[index]
    sum.y += vel.y[index]
    #println(vel.x[index], '\t', vel.y[index])

    # Extending upwards
    arc = Position(x, round((cir.y-cir.radius)/vel.dy),0)
    index = Int64(arc.y + arc.x * vel.yDim)
    index = index_check(index, length(vel.x))
    #println("circle y and rad are: ", cir.y, '\t', cir.radius, '\t', index)
    sum.x += vel.x[index]
    sum.y += vel.y[index]
    #println(vel.x[index], '\t', vel.y[index])

    return sqrt(sum.x*sum.x + sum.y*sum.y)
    
end

# Initialization function for circles for superfluid vortex locating
function init(num::Int64, box_length::Float64, max_rad::Float64, 
              min_rad::Float64)

    circles = Circle[]

    for i = 1:num
        rad = rand() * (max_rad - min_rad) + min_rad
        x = rand() * (box_length - 3*rad) + 1.5*rad
        y = rand() * (box_length - 3*rad) + 1.5*rad
        push!(circles,Circle(x, y, rad))
    end

    return circles
    
end

# function for tournament selection of circles
function tournament(population::Vector{Circle}, vel::Field, size::Int64)

    # Selecting the first individual
    best = rand(population)

    # Choosing best from random selection of others
    for j = 2:size
        candidate = rand(population)
        if (fitness(candidate, vel) < fitness(best, vel))
            best = candidate
        end
    end

    return best
end

# Tournament selection, size is tournament size, and prob is prob of selection
# num_parents must be even
# The num_children = num_parents, make sure that 
#     the children size < population size - elite number
function selection(population::Vector{Circle}, size::Int64, vel::Field,
                   num_parents::Int64)

    # Forcing a multiple of 2 for number of parents
    if (num_parents % 2 != 0)
        num_parents += 1
    end

    # result is the final selected population
    parents = fill(Circle(0.0,0.0,0.0), num_parents)

    # selection loop
    for i = 1:num_parents
        parents[i] = tournament(population, vel, size)
    end

    return parents
end

# Function to enforce circles being in box
function radius_check(cir::Circle, box_length::Float64, min_rad::Float64)
    if (cir.radius < min_rad)
        cir.radius = min_rad
    end
    diff_x = cir.x - cir.radius
    if (diff_x < 0)
        cir.x -= diff_x
    end 
    diff_x = cir.x + cir.radius - box_length
    if (diff_x > 0)
        cir.x -= diff_x
    end

    diff_y = cir.y - cir.radius 
    if (diff_y < 0)
        cir.y -= diff_y
    end 
    diff_y = cir.y + cir.radius - box_length
    if (diff_y > 0)
        cir.y -= diff_y
    end

    return cir

end

# Crossover event for circles in a superfluid
# Note: There are only 3 variables to twiddle for 2d: x,y,radius
#       This means we should use an alpha-blend crossover event
function crossover(parents::Vector{Circle}, cross_rate::Float64, 
                   box_length::Float64, min_rad::Float64)

    # initializing a vector for all children of length(parents) in size
    offspring = fill(Circle(0.0,0.0,0.0), length(parents))
    child1 = Circle(0.0,0.0,0.0)
    child2 = Circle(0.0,0.0,0.0)

    # Selecting 2 parents -> 2 children
    for i = 1:2:length(parents)
        child1 = Circle(0.0,0.0,0.0)
        child2 = Circle(0.0,0.0,0.0)
        
        # Selecting random numbers to see whether we crossover x, y, or rad
        index_x = rand()
        index_y = rand()
        index_rad = rand()

        # Actually crossing over each variable x, y, radius
        if index_x > cross_rate
            cross_loc = rand()
            child1.x = parents[i].x * cross_loc 
                       + parents[i+1].x * (1 - cross_loc)
            child2.x = parents[i+1].x * cross_loc 
                       + parents[i].x * (1 - cross_loc)
        else
            child1.x = parents[i].x
            child2.x = parents[i+1].x
        end
        if index_y > cross_rate
            cross_loc = rand()
            child1.y = parents[i].y * cross_loc 
                       + parents[i+1].y * (1 - cross_loc)
            child2.y = parents[i+1].y * cross_loc 
                       + parents[i].y * (1 - cross_loc)
        else
            child1.y = parents[i].y
            child2.y = parents[i+1].y
        end
        if index_rad > cross_rate
            cross_loc = rand()
            child1.radius = parents[i].radius * cross_loc 
                            + parents[i+1].radius * (1 - cross_loc)
            child2.radius = parents[i+1].radius * cross_loc 
                            + parents[i].radius * (1 - cross_loc)
        else
            child1.radius = parents[i].radius
            child2.radius = parents[i+1].radius
        end
        #child1 = radius_check(child1, box_length, min_rad)
        #child2 = radius_check(child2, box_length, min_rad)
        offspring[i] = child1
        offspring[i+1] = child2

    end

    return offspring
end

# Selection of individuals for repopulation
function repopulate(population::Vector{Circle}, parents::Vector{Circle},
                    offspring::Vector{Circle}, vel::Field, elite::Int64)

    # sorting the vector to grab the elite
    sort!(population; lt = (a, b) -> fitness(a, vel) < fitness(b, vel))

    # replace all non-elites
    for i = 1:length(population)-elite
        population[i + elite] = offspring[i]
    end

    return population

end

# Function for the mutation of an individual
function mutation(offspring::Vector{Circle}, mutation_rate::Float64,
                  box_length::Float64, max_rad::Float64, min_rad::Float64)

    # going through all offspring and mutating them if necesary
    for i = 1:length(offspring)
        # randomly shuffling the radius first
        if rand() < mutation_rate
            offspring[i].radius = rand() * (max_rad - min_rad) + min_rad
        end

        # Now for x and y
        if rand() < mutation_rate
            offspring[i].x = rand() * (box_length - 3*offspring[i].radius)
                             + 1.5 * offspring[i].radius 
        end
        if rand() < mutation_rate
            offspring[i].y = rand() * (box_length - 3*offspring[i].radius)
                             + 1.5 * offspring[i].radius 
        end
    end

    return offspring
end

# main function
function main(timesteps::Int64)
    num = 100
    box_length = 1.0
    max_rad = 0.1
    x_loc = fill(0.0, num)

    # creating our 2d vel field for vortex finding
    xDim = 128
    yDim = 128
    zDim = 1

    dx = 1 / xDim
    dy = 1 / yDim
    dz = 0.0
    min_rad = dx

    vortices = [Position(0.5,0.5,0)]
    vel = find_field2d(vortices, xDim, yDim, dx, dy)
    population::Vector{Circle} = init(num, box_length, max_rad, min_rad)
    for i = 1:length(population)
        x_loc[i] = population[i].x
    end
    plot(x_loc)

    for i =1:timesteps
        mutation_rate = .5 - 0.5*(Float64(i)/Float64(timesteps))
        println("mutation rate is: ", mutation_rate)
        parents::Vector{Circle} = selection(population, 10, vel, num)
        offspring::Vector{Circle} = crossover(parents, 1.0, box_length, min_rad)
        offspring = mutation(offspring, mutation_rate, box_length, max_rad, min_rad)
        population = repopulate(population, parents, offspring, vel, 10)
        for i = 1:length(population)
            x_loc[i] = population[i].x
        end
        plot(x_loc)
        sleep(1)
        clf()
    end
end

function circle_test()
    xDim = 64
    yDim = 64
    zDim = 1

    dx = 1 / xDim
    dy = 1 / yDim
    dz = 0.0

    vortices = [Position(0.5,0.5,0)]

    vel = find_field2d(vortices, xDim, yDim, dx, dy)

    # Creating circle to check fitness
    cir = Circle(0.5,0.5, 0.1)

    fit = fitness(cir, vel)
    println(fit)

    circles = init(10, 10.0, 5.0)

    for cir in circles
        println(cir.x, '\t', cir.y, '\t', cir.radius)
    end

end

#circle_test()

main(50)
