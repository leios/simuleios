#-------------split_op.jl------------------------------------------------------#
#
# Purpose: To create a quick simulation of a quantum system
#
#   Notes: Incorporate imaginary time into simulation function
#          Figure out why simulation is blank every other timestep
#
#------------------------------------------------------------------------------#
using Plots
pyplot()

# struct to hold all parameters for simulation
struct Param
    xmax::Float64
    res::Int64
    dt::Float64
    timesteps::Int64
    dx::Float64
    x::Vector{Float64}
    dk::Float64
    k::Vector{Float64}
    im_time::Bool

    Param() = new(10.0, 512, 0.05, 1000, 2 * 10.0/512,
                  Vector{Float64}(-10.0 + 10.0/512 : 20.0/512 : 10.0),
                  pi / 10.0,
                  Vector{Float64}(vcat(0:512/2 - 1, -512/2 : -1) * pi/10.0),
                  false)
    Param(xmax::Float64, res::Int64, dt::Float64, timesteps::Int64,
          im_val::Bool) = new(
              xmax, res, dt, timesteps,
              2*xmax/res, Vector{Float64}(-xmax+xmax/res:2*xmax/res:xmax),
              pi/xmax, Vector{Float64}(vcat(0:res/2-1, -res/2:-1)*pi/xmax),
              im_val
          )
end

# struct to hold all operators
mutable struct Operators
    V::Vector{Complex{Float64}}
    PE::Vector{Complex{Float64}}
    KE::Vector{Complex{Float64}}
    wfc::Vector{Complex{Float64}}

    Operators(res) = new(Vector{Complex{Float64}}(res),
                         Vector{Complex{Float64}}(res),
                         Vector{Complex{Float64}}(res),
                         Vector{Complex{Float64}}(res))
end

# Function to update V every timestep
function update_V(par::Param, opr::Operators, voffset::Float64)
    thresh = 5
    V1 = 0.5 * (par.x - voffset).^2 + 4 
    V2 = 4*(2*(par.x + voffset)).^2
    for i = 1:length(V1)
        if (V1[i] > thresh)
            V1[i] = thresh
        end
        if (V2[i] > thresh)
            V2[i] = thresh
        end
    end

    opr.V = V1 + V2
    if (par.im_time)
        opr.PE = exp.(-0.5*opr.V*par.dt)
    else
        opr.PE = exp.(-im*0.5*opr.V*par.dt)
    end
end

# Function to initialize the wfc and potential
function init(par::Param, voffset::Float64, wfcoffset::Float64)
    opr = Operators(length(par.x))
    update_V(par, opr, voffset)
    opr.wfc = 5 * exp.(-(par.x - wfcoffset).^2/2)
    if (par.im_time)
        opr.KE = exp.(-0.5*par.k.^2*par.dt)
    else
        opr.KE = exp.(-im*0.5*par.k.^2*par.dt)
    end

    return opr
end

# Function for the split-operator loop
function split_op(par::Param, opr::Operators)

    for i = 1:par.timesteps
        # Half-step in real space
        opr.wfc = opr.wfc .* opr.PE

        # fft to momentum space
        opr.wfc = fft(opr.wfc)

        # Full step in momentum space
        opr.wfc = opr.wfc .* opr.KE

        # ifft back
        opr.wfc = ifft(opr.wfc)

        # final half-step in real space
        opr.wfc = opr.wfc .* opr.PE

        # density for plotting and potential
        density = abs2.(opr.wfc)

        # renormalizing for imaginary time
        if (par.im_time)
            sum = 0
            for element in density
                sum += element
            end
            sum /= (par.res*2)
            println(sum)

            for j = 1:length(opr.wfc)
                opr.wfc[j] /= sqrt(sum)
            end
        end

#=
        # outputting wavefunction to file
        f = open("wfc" * string(i), "w")
        for element in density
            write(f, string(element)*'\n')
        end
        close(f)
=#

        if ((i-1) % div(par.timesteps, 100) == 0)
            plot([density, real(opr.V)])
            savefig("density" * string(lpad(i-1, 8, 0)) * ".png")
            println(i)
        end

#=
        if (i <= par.timesteps)
            update_V(par, opr, -5 + 10*i/par.timesteps)
        end
=#
    end
end

# main function
function main()
    par = Param(10.0, 512, 0.05, 100, true)
    opr = init(par, -1.00, -1.00)
    split_op(par, opr)
end

main()
