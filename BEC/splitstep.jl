#=-------------splitstep.jl----------------------------------------------------#
#
#              split step method for macroscopic wave function generation
#
# Purpose: This file is intended to create a splitstep method for BEC systems.
#          Eventually we may incorporate real-time evolution, but we need to 
#          first generate the grounds state and go from there.
#
#   Notes: Units of hbar = c  = 1
#-----------------------------------------------------------------------------=#

global xmax = 20
global res = 2^8
global g = 500

# This initializes the potential well and also the gaussian init wavefunction
function initialize()
    gaus = zeros(res)
    pot = zeros(res)
    for ix = 1:res
        x = ix * (xmax/res) - xmax / 2
        gaus[ix] = exp(-x * x )
        pot[ix] = 0.5 * x * x
        #println(gaus[ix], '\t', pot[ix], '\t', x)
    end
    return gaus, pot
end

# This will return the kinetic and potential energies at every timestep
# Note unphysical units will be fixed in the future.
function energy(wave, pot, dt)

    KE = complex(zeros(size(wave,1)))
    PE = complex(zeros(size(wave,1)))

    dk = 2pi / xmax

    for i = 1:size(wave,1)
        if i <= size(wave,1) / 2
            k = dk * (i)
        else
            k = dk * (i - (size(wave,1) / 2))
        end
        KE[i] = exp( -0.5 * (k*k) * dt)
        PE[i] = exp( -0.5 * (pot[i] + g*wave[i]*wave[i]) *dt)
    end
    return PE, KE
end

# The whole shebang
# Psi * Uv -> fft -> Psi * Uk -> ifft -> renormalization -> cycle
function splitstep(stepnum, dt)

    output = open("out.dat", "w")
    wave, pot = initialize()

    for j = 1:stepnum

        PE, KE = energy(wave, pot, dt)
        norm_const = 0

        for i = 1:res
            wave[i] *= PE[i]
        end
 
        wave = fft(wave)

        for i = 1:res
            wave[i] *= KE[i]
        end

        wave = ifft(wave)

        norm_const = 0
        for i = 1:res
            norm_const += abs2(wave[i])
        end

        println(norm_const)

        for i = 1:res
            wave[i] *= 1/norm_const
        end

        if j % 1000 == 0 || j == 0
            for i = 1:res
                println(output, wave[i])
            end

            print(output, '\n', '\n')
        end

    end
end

# Main

splitstep(10000, 0.001)
