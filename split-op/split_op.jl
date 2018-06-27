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

    Param() = new(10.0, 512, 0.05, 1000, 2 * 10.0/512,
                  Vector{Float64}(-10.0 + 10.0/512 : 20.0/512 : 10.0),
                   pi / 10.0,
                  Vector{Float64}(vcat(0:512/2 - 1, -512/2 : -1) * pi/10.0))
    Param(xmax::Float64, res::Int64, dt::Float64, timesteps::Int64) = new(
              xmax, res, dt, timesteps,
              2*xmax/res, Vector{Float64}(-xmax+xmax/res:2*xmax/res:xmax),
              pi/xmax, Vector{Float64}(vcat(0:res/2-1, -res/2:-1)*pi/xmax)
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
    V2 = 2*(par.x + voffset).^2
    for i = 1:length(V1)
        if (V1[i] > thresh)
            V1[i] = thresh
        end
        if (V2[i] > thresh)
            V2[i] = thresh
        end
    end

    opr.V = V1 + V2
    opr.PE = exp.(-0.5*im*opr.V*par.dt)
end

# Function to initialize the wfc and potential
function init(par::Param, voffset::Float64, wfcoffset::Float64)
    opr = Operators(length(par.x))
    update_V(par, opr, voffset)
    opr.wfc = 3 * exp.(-(par.x - wfcoffset).^2/2)
    opr.KE = exp.(-0.5*im*par.k.^2*par.dt)

    return opr
end

# Function for the split-operator loop
function split_op(par::Param, opr::Operators)

    for i = 1:par.timesteps
        # Half-step in real space
        opr.wfc = opr.wfc .* opr.PE

        # fft to phase space
        opr.wfc = fft(opr.wfc)

        # Full step in phase space
        opr.wfc = opr.wfc .* opr.KE

        # ifft back
        opr.wfc = ifft(opr.wfc)

        # final half-step in real space
        opr.wfc = opr.wfc .* opr.PE

        # plotting density and potential
        density = abs2.(opr.wfc)

        plot([density, real(opr.V)])
        savefig("density" * string(lpad(i, 5, 0)) * ".png")

        if (i <= 1000)
            update_V(par, opr, -5 + 10*i/1000.)
        end
        println(i)
    end
end

# main function
function main()
    par = Param(10.0, 512, 0.05, 1000)
    opr = init(par, -5.0, -5.0)
    split_op(par, opr)
end

main()
