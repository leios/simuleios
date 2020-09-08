using Images, ImageView

@enum Direction vertical=1 horizontal=2

struct Seam
    direction::Direction
    elements::Vector{Int}
end

# function to return magnitude of image elements
# Todo: Maybe convert to another color-space for better magnitude
function image_magnitude(img_element)
    return sqrt(img_element.r^2 + img_element.g^2 + img_element.b^2)
end

# function to find energy of image
function find_energy(img)
    return abs.(imfilter(image_magnitude.(img), Kernel.sobel().*10))
end

# function to create seams and return seam of minimum energy
# n is number of seams
function find_seam(energy, n, direction)
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

# This function removes a pixel from an image and saves 2 images:
#     one with the pixel darkened
#     another with the entire row of pixels shifted
function color_pixel!(cimg, pixel::CartesianIndex, filebase, iteration)

    # Image 1: setting the pixel black
    println(cimg[pixel], typeof(cimg))
    cimg[pixel] = RGB(1,1,1)

    save(filebase*lpad(string(iteration-1),5,string(0))*".jpg", cimg)
end

function remove_pixel!(img, pixel::CartesianIndex, filebase, iteration)

    # Image 2: shift the image
    if pixel[2] > 1 && pixel[2] < size(img)[2]
        img[pixel[1],:] = vcat(img[pixel[1], 1:pixel[2]-1],
                               img[pixel[1], pixel[2]+1:end],
                               RGB(0,0,0))
    elseif pixel[2] == 1
        img[pixel[1],:] = vcat(img[pixel[1],2:end], RGB(0,0,0))
    elseif pixel[2] == size(img)[2]
        img[pixel[1],:] = vcat(img[pixel[1],1:end-1], RGB(0,0,0))
    end

    save(filebase*lpad(string(iteration-1),5,string(0))*".jpg", img)

end

function naive_removal(img, n)
    energy = find_energy(img)
    colored_img = copy(img)

    pixels = Array{CartesianIndex, 1}(undef,n)

    for i = 1:n
        pixel = argmin(energy)
        colored_img[pixel] = RGB(0,0,0)
        save("cout"*lpad(string(i-1),5,string(0))*".jpg", colored_img)

        pixels[i] = pixel
        energy[pixel] = 1000

    end

    removed_img = copy(colored_img)

    for i = 1:n
        if pixels[i][2] > 1 && pixels[i][2] < size(removed_img)[2]
            removed_img[pixels[i][1],:] = vcat(removed_img[pixels[i][1], 1:pixels[i][2]-1],
                                               removed_img[pixels[i][1], pixels[i][2]+1:end],
                                               RGB(0,0,0))
        elseif pixels[i][2] == 1
            removed_img[pixels[i][1],:] = vcat(removed_img[pixels[i][1],2:end],
                                               RGB(0,0,0))
        elseif pixels[i][2] == size(img)[2]
            removed_img[pixels[i][1],:] = vcat(removed_img[pixels[i][1],1:end-1], RGB(0,0,0))
        end

        save("rout"*lpad(string(i-1),5,string(0))*".jpg", removed_img)

    end
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
