using Images, ImageView

@enum Direction vertical=1 horizontal=2

struct Seam
    direction::Direction
    elements::Vector{Int}
end

# function to return magnitude of image elements
# Todo: Maybe convert to another color-space for better magnitude
function brightness(img_element)
    return img_element.r + img_element.g + img_element.b
end

# function to find energy of image
function find_energy(img)
    energy_x = imfilter(brightness.(img), Kernel.sobel()[1])
    energy_y = imfilter(brightness.(img), Kernel.sobel()[2])
    return sqrt.(energy_x.^2 + energy_y.^2)
end

# function to find direction to move in
# TODO: generalize to horizontal and vertical seams
function find_energy_map(energy)
    energy_map = zeros(size(energy))
    energy_map[end,:] .= energy[end,:]
    next_elements = zeros(Int, size(energy))

    for i = size(energy)[1]-1:-1:1
        for j = 1:size(energy)[2]
            left = max(1, j-1)
            right = min(size(energy)[2], j+1)

            local_energy, next_element = findmin(energy_map[i+1, left:right])
            energy_map[i,j] += local_energy + energy[i,j]
            next_elements[i,j] = next_element - 2
        end
    end

    return energy_map, next_elements
end

# function to create seams and return seam of minimum energy
# n is number of seams
function find_seam(energy, direction)

    energy_map, next_elements = find_energy_map(energy)

    seam = Seam(direction, zeros(size(energy[direction])))

    
end

# function to remove seams
function remove_seam(img, seam::Seam)
    img_res = size(img)
    # TODO: Make this pretty
    # img_res[Int(seam.direction)] -= 1
    if seam.direction == horizontal
        img_res = (img_res[1], img_res[2]-1)
    elseif seam.direction == vertical
        img_res = (img_res[1]-1, img_res[2])
    else
        error("direction not found!")
    end

    # preallocate image
    new_img = Array{RGB}(undef, img_res)

    for i = 1:length(seam.elements)
        if seam.direction == horizontal
            # TODO: figure out hcat here
            new_img[i, :] = hcat(img[i, 1:seam.elements[i]-1], 
                                 img[i, seam.elements[i]+1:end])
        else
            new_img[:, i] = vcat(img[1:seam.elements[i]-1, i], 
                                 img[seam.elements[i]+1:end, i])
        end
    end

    return new_img
end

# Putting it all together
function seam_carving(img, res, n)

    # vertical
    while size(img)[1] != res[1]
        energy = find_energy(img)
        seam = find_seam(energy, n, vertical)
        img = remove_seam(img, seam)
    end

    # horizontal
    while size(img)[2] != res[2]
        energy = find_energy(img)
        seam = find_seam(energy, n, horizontal)
        img = remove_seam(img, seam)
    end
end
