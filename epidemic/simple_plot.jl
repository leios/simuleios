# Links: https://github.com/JuliaGraphs/GraphPlot.jl#arguments
# https://docs.juliaplots.org/latest/graphrecipes/examples/#Undirected-graph
using GraphRecipes
using LightGraphs
import Plots
Plots.pyplot()

h = watts_strogatz(50, 6, 0.3)

graphplot(h)
