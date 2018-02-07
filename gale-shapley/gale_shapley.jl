#------------gale_shapley.jl---------------------------------------------------#
#
# Purpose: To implement a solution to the stable Marriage problem
#
#------------------------------------------------------------------------------#

using DataStructures

function find_preference(n::Int)
    men_pref = fill(-1, n, n)
    women_pref = fill(-1, n, n)
    for i = 1:n
        men_pref[i,:] = shuffle(Vector(1:n))
        women_pref[i,:] = shuffle(Vector(1:n))
    end

    return (men_pref, women_pref)
end

function propose(wid::Int, mid::Int, woman::Array{Int,1}, pairs::Vector{Int})
    mid_0 = find(x -> x == wid, pairs)
    if(length(mid_0) != 0)
        rank_0 = woman[mid_0]
        rank_1 = woman[mid]
        if (rank_1[1] > rank_0[1])
            pairs[mid] = wid
        end
    else
        pairs[mid] = wid
        println(pairs)
    end
end

function gale_shapley(men::Array{Int, 2}, women::Array{Int, 2})
    n = Int(sqrt(length(men)))
    pairs = Vector{Int}(fill(-1.0, n))

    for i = 1:n
        for j = 1:n
            propose(men[j,i], j, women[j,:], pairs)
        end
    end

    return pairs
end

function main(n::Int)
    (men, women) = find_preference(n)
    pairs = gale_shapley(men, women)

    println(pairs)
end

main(10)
