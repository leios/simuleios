#=-------------splitstep.jl----------------------------------------------------#
#
#              split step method for macroscopic wave function generation
#
# Purpose: This file is intended to create a splitstep method for BEC systems.
#          Eventually we may incorporate real-time evolution, but we need to 
#          first generate the grounds state and go from there.
#
#-----------------------------------------------------------------------------=#

# This initializes the potential well and also the gaussian init wavefunction
function initialize(res)
    gaus = [Float64(0) for i = 1:res]
    pot = [Float64(0) for i = 1:res]
    for dx = 1:res
        x = dx * (5/res) - 2.5
        gaus[dx] = exp(-x * x)
        pot[dx] = x * x
        #println(gaus[dx], '\t', pot[dx], '\t', x)
    end
    return gaus, pot
end

# This will return the kinetic and potential energies at every timestep
function energy(wave, pot, dt)
    #ficticious g for now
    g = 1
    PE = exp( -(pot + g * wave .* wave) * dt)
    # KE relies on k, not yet determined
    KE = exp( -dt)
    return PE, KE
end

# The whole shebang
function splitstep(res)
    wave, pot = initialize(res)
    PE, KE = energy(wave, pot, 0.001)
end

# Main
splitstep(1000)
