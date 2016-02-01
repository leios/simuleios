#=-------------splitstep_units.jl----------------------------------------------#
#
#              split step method for macroscopic wave function generation
#
# Purpose: This file is intended to create a splitstep method for BEC systems.
#          Eventually we may incorporate real-time evolution, but we need to 
#          first generate the grounds state and go from there.
#
#   Notes: We are working with Rubidium 87 for this simulation, and attempting 
#          to use appropriate units
#-----------------------------------------------------------------------------=#

global hbar = 1.055E-34
global scatl = 4.67E-9
global boson_num = 1E5
global mass = 1.4431607E-25
global coupling = 4 * pi * hbar * hbar * scatl * boson_num / mass
global radius = sqrt(hbar / (2 * mass))
global R = 15^(0.2) * (boson_num * scatl * sqrt(mass / hbar))^0.2
global xmax = 6 * R * radius * 0.000000000005

# This initializes the potential well and also the gaussian init wavefunction
function initialize(res, dt)
    gaus = complex(zeros(res))
    pot = zeros(res)
    for ix = 1:res
        x = complex(ix * (xmax/res) - xmax / 2)
        gaus[ix] = exp(-0.5 * mass * x * x * dt/ (2 * hbar)) + 0im
        pot[ix] = radius * x * x
        #println(gaus[ix], '\t', pot[ix], '\t', x)
        #println(gaus[ix])
    end
    return gaus, pot
end

# This will return the kinetic and potential energies at every timestep
# Note unphysical units will be fixed in the future.
function energy(wave, pot, dt, res)

    #ficticious g for now
    PE = zeros(size(wave,1))
    KE = zeros(size(wave,1))
    dk = 2pi / xmax

    for i = 1:size(wave,1)

        k = dk * (i - (size(wave,1) / 2))

        PE[i] = exp( -(pot[i] + coupling * abs2(wave[i])) * dt/(hbar))

        # KE relies on k, not yet determined
        KE[i] = exp( -hbar * hbar * (k*k)/(2*mass) * dt)
    end
    return PE, KE
end

# The whole shebang
# Psi * Uv -> fft -> Psi * Uk -> ifft -> renormalization -> cycle
function splitstep(res)

    output = open("out.dat", "w")
    wave, pot = initialize(res, 0.1)

    #println(wave)

    for j = 1:10000

        PE, KE = energy(wave, pot, 0.1, res)
 
        for i = 1:res
            wave[i] *= PE[i]
        end
 
        wave = fft(wave)

        for i = 1:res
            wave[i] *= KE[i]
        end

        wave = ifft(wave)

        norm_const = 0.0
        for i = 1:res
            norm_const += abs2(wave[i])
        end

        #println(norm_const)

        wave *= 1/norm_const

        if j % 1000 == 0
            for i = 1:res
                println(output, real(wave[i]))
            end

            print(output, '\n', '\n')
        end

    end
end

# Main

splitstep(512)
println(hbar, '\t', scatl, '\t', boson_num, '\t', mass, '\t', coupling, '\t',
        radius, '\t', R, '\t', xmax)
