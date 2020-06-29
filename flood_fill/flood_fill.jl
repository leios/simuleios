using Plots

function create_circle!(canvas, loc::CartesianIndex, radius, threshold,
                        fill_val)
    for i = 1:size(canvas)[1]
        for j = 1:size(canvas)[2]
            if ((radius-threshold)^2
                < (i-loc[1])^2+(j-loc[2])^2
                < (radius+threshold)^2)
                canvas[i,j] = fill_val
            end
        end
    end
end

function recursive_fill!(canvas, loc::CartesianIndex,
                         fill_val, replacement_val)
    # bounds check
    #if min(loc - CartesianIndex(size(canvas))) < 1
    #    return
    #end

    if (fill_val == replacement_val)
        return
    elseif (canvas[loc] != fill_val)
        return
    else
        canvas[loc] = replacement_val
    end

    recursive_fill!(canvas, loc + CartesianIndex(0, 1), fill_val,
                    replacement_val)
    recursive_fill!(canvas, loc + CartesianIndex(0, -1), fill_val,
                    replacement_val)
    recursive_fill!(canvas, loc + CartesianIndex(1, 0), fill_val,
                    replacement_val)
    recursive_fill!(canvas, loc + CartesianIndex(-1, 0), fill_val,
                    replacement_val)
end

function main()
    canvas = zeros(100,100)
    loc = CartesianIndex(50,50)
    create_circle!(canvas, loc, 30, 5, 0.75)
    recursive_fill!(canvas, loc, 0.0, 0.5)

    heatmap(canvas)
end
main()
