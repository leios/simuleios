#-------------barnsley_fern.jl-------------------------------------------------#
#
# Purpose: quick test of the barnsel fern iterated function system
#
#------------------------------------------------------------------------------#

using PyPlot

function barnsley_fern(n::Int64)
    point = [0.0, 0.0]
    for i = 1:n
        scatter(point[1], point[2], 1, 1)
        rnd = rand()
        if (rnd <= 0.01)
            point = [0 0; 0 0.16]*point
        elseif(rnd > 0.01 && rnd <= 0.86)
            point = [0.85 0.04; -0.04 0.85]*point + [0, 1.60]
        elseif(rnd > 0.86 && rnd <= 0.93)
            point = [0.2 -0.26; 0.23 0.22]*point + [0, 1.60]
        else
            point = [-0.15 0.28; 0.26 0.24]*point + [0, 0.44]
        end
    end
    savefig("yo.png")
end

barnsley_fern(10000)
