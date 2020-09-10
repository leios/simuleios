using Images, ImageView

function draw_seam(img, seam)
    img_w_seam = copy(img)
    for i = 1:size(img)[1]
        img_w_seam[i,seam[i]] = RGB(1,1,1)
    end 

    return img_w_seam
end

function write_image(img, i; filebase = "out")
    save(filebase*lpad(string(i),5,string(0))*".png", img)
end

# function to return magnitude of image elements
function brightness(img_element)
    return img_element.r + img_element.g + img_element.b
end

# function to find energy of image
function find_energy(img)
    energy_x = imfilter(brightness.(img), Kernel.sobel()[2])
    energy_y = imfilter(brightness.(img), Kernel.sobel()[1])
    return sqrt.(energy_x.^2 + energy_y.^2)
    return abs2.(imfilter(brightness.(img), Kernel.sobel()))
end

# function to find direction to move in
function find_energy_map(energy)
    energy_map = zeros(size(energy))
    energy_map[end,:] .= energy[end,:]
    next_elements = zeros(Int, size(energy))

    for i = size(energy)[1]-1:-1:1
        for j = 1:size(energy)[2]
            left = max(1, j-1)
            right = min(size(energy)[2], j+1)
            if left < 1
                println("less than 1")
            end

            local_energy, next_element = findmin(energy_map[i+1, left:right])
            energy_map[i,j] += local_energy + energy[i,j]
            next_elements[i,j] = next_element -2

            # correct for only having 2 options on left edge.
            if left == 1 && size(energy)[2] > 3
                next_elements[i,j] += 1
            end
        end
    end

    return energy_map, next_elements
end

function generate_seam(energy, next_elements, element)
    seam = zeros(Int, size(next_elements)[1])
    seam[1] = element

    seam_energy = energy[element]

    for i = 2:length(seam)
        seam[i] = seam[i-1] + next_elements[i, seam[i-1]]
        seam_energy += energy[i, seam[i]]
    end

    return seam, seam_energy
end


# function to create seams and return seam of minimum energy
# n is number of seams
function find_seam(energy)

    energy_map, next_elements = find_energy_map(energy)

    kept_energy = Inf
    kept_seam = zeros(size(energy)[1])

    for i = 1:size(energy)[2]
        seam, seam_energy = generate_seam(energy, next_elements, i)
        if seam_energy < kept_energy
            kept_seam = seam
            kept_energy = seam_energy
        end
    end

    return kept_seam, kept_energy
end

# function to remove seams
function remove_seam(img, seam)
    img_res = (size(img)[1], size(img)[2]-1)

    # preallocate image
    new_img = Array{RGB}(undef, img_res)

    for i = 1:length(seam)
        if i > 1 && i < size(img)[2]
            new_img[i, :] .= vcat(img[i, 1:seam[i]-1], 
                                  img[i, seam[i]+1:end])
        elseif i == 1
            new_img[i,:] .= img[i,2:end]
        elseif i == size(img)[2]
            new_img[i,:] .= img[i,1:end-1]
        end
    end

    return new_img
end

# Putting it all together
function seam_carving(img, res)

    if res < 0 || res > size(img)[2]
        error("resolution not acceptable")
    end

    for i = (1:size(img)[2] - res)
        energy = find_energy(img)
        seam, energy = find_seam(energy)
        img = remove_seam(img, seam)
        write_image(img, res, i)
    end
end
