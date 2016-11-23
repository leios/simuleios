#-------------vortex.jl--------------------------------------------------------#
#
# Purpose: Find a field on the outside of a superfluid vortex
#
#------------------------------------------------------------------------------#

# Field type will be used for velocity field around vortex
type Field
   xDim::Int64
   yDim::Int64
   zDim::Int64

   dx::Float64
   dy::Float64
   dz::Float64

   # x, y, and z are all of size xDim*yDim*zDim
   x::Array{Float64, 1}
   y::Array{Float64, 1}
   z::Array{Float64, 1}
   
end

# position type will be used to find the location of vortices
type Position
    x::Float64
    y::Float64
    z::Float64
end

# function to initialize our field
function init_field(xDim::Int64, yDim::Int64, zDim::Int64, 
                    dx::Float64, dy::Float64, dz::Float64)
    gsize = xDim * yDim * zDim
    empty_x = fill(0.0, gsize)
    empty_y = fill(0.0, gsize)
    empty_z = fill(0.0, gsize)
    return Field(xDim, yDim, zDim, dx, dy, dz, 
                 empty_x, empty_y, empty_z)
end

# Function to find distance between two positions
function distance(loc1::Position, loc2::Position)
    x = loc1.x - loc2.x
    y = loc1.y - loc2.y
    z = loc1.z - loc2.z

    return sqrt(x*x + y*y + z*z)
end

# Function for the dot product between two positions
function dot(loc1::Position, loc2::Position)
    return loc1.x*loc2.x + loc1.y*loc2.y + loc1.z*loc2.z
end

# Function to find the magnitude of a position
function mag(loc::Position)
    return sqrt(loc.x*loc.x + loc.y*loc.y + loc.z*loc.z)
end

# Function to find the angle between two positions
function anglexy(loc1::Position, loc2::Position, radius)
    x = loc2.x - loc1.x
    y = loc2.y - loc1.y
    #return acos(dot(loc1,loc2)/(mag(loc1)*mag(loc2)))

    angle = 0
    if (sign(x) <= 0 && sign(y) > 0)
        angle = atan(-x/y) + 0.5*pi
    elseif (sign(x) < 0 && sign(y) <= 0)
        angle = atan(y/x) + pi
    elseif (sign(x) >= 0 && sign(y) < 0)
        angle = atan(-x/y) + 1.5 * pi
    else
        angle = atan(y/x)
    end
    return angle

end

# Function to find velocity around vortex in 2D
function find_field2d(location::Vector{Position}, xDim::Int64, yDim::Int64, 
                      dx::Float64, dy::Float64)

    # initialize the field
    vel = init_field(xDim, yDim, 1, dx, dy, 0.0)

    # index is for internal keeping of the indices
    index = 0

    for i = 0:xDim - 1
        for j = 1:yDim
            index = j + i * yDim

            # Going through all vortex locations to find velocity field
            for k = 1:length(location)
                # First find the distance and angle to the vortex center
                point = Position(Float64((i)) * dx, Float64(j - 1)*dy, 0)
                radius = distance(point, location[k])
                theta = anglexy(point, location[k], radius)
                println(theta)
                vel.x[index] = (1/(2pi * radius)) * cos(theta)
                vel.y[index] = (1/(2pi * radius)) * sin(theta)
            end
        end
    end

    return vel
end

# function to output data in a gnuplot fashion
function output2d(vel::Field)

    # opening files
    xfile = open("x.dat", "w")
    yfile = open("y.dat", "w")

    for i = 0:vel.xDim - 1
        for j = 1:vel.yDim
            index = j + i * vel.yDim
            x = Float64((i+1)) * vel.dx
            y = Float64(j) * vel.dy
            write(xfile, "$x\t$y\t$(vel.x[index])\n")
            write(yfile, "$x\t$y\t$(vel.y[index])\n")
        end
    end

    close(xfile)
    close(yfile)
end

# main function
function main()
    xDim = 64
    yDim = 64
    zDim = 1

    dx = 1 / xDim
    dy = 1 / yDim
    dz = 0.0

    vortices = [Position(0.5,0.5,0)]

    vel = find_field2d(vortices, xDim, yDim, dx, dy)

    output2d(vel,)
end

main()
