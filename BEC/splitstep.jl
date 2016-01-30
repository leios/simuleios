#=-------------splitstep.jl----------------------------------------------------#
#
#              split step method for macroscopic wave function generation
#
# Purpose: This file is intended to create a splitstep method for BEC systems.
#          Eventually we may incorporate real-time evolution, but we need to 
#          first generate the grounds state and go from there.
#
#   Notes: We are working with Rubidium 87 for this simulation
#-----------------------------------------------------------------------------=#

global hbar = 1.055E-34
global scatl = 4.67E-9
global boson_num = 1E5
global mass = 1.4431607E-25
global coupling = 4 * pi * hbar * hbar * scatl * boson_num / mass
global radius = sqrt(hbar / (2 * mass))
global R = 15^(0.2) * (boson_num * scatl * sqrt(mass / hbar))^0.2
global xmax = 6 * R * radius * 0.00000000005

# This initializes the potential well and also the gaussian init wavefunction
function initialize(res)
    gaus = zeros(res)
    pot = zeros(res)
    for dx = 1:res
        x = dx * (xmax/res) - xmax / 2
        gaus[dx] = exp(-x * x / (R * radius))
        pot[dx] = radius * x * x
        #println(gaus[dx], '\t', pot[dx], '\t', x)
    end
    return gaus, pot
end

# This will return the kinetic and potential energies at every timestep
# Note unphysical units will be fixed in the future.
function energy(wave, pot, dt, res)

    #ficticious g for now
    PE = zeros(size(wave,1))
    KE = zeros(size(wave,1))
    dk = 2pi / res

    for i = 1:size(wave,1)

        k = dk * (i - (size(wave,1) / 2))

        PE[i] = exp( -(pot[i] + coupling * wave[i]*wave[i]) * dt/hbar)

        # KE relies on k, not yet determined
        KE[i] = exp( -hbar*hbar*(k*k)/(2*mass) * dt)
    end
    return PE, KE
end

# The whole shebang
# Psi * Uv -> fft -> Psi * Uk -> ifft -> renormalization -> cycle
function splitstep(res)

    output = open("out.dat", "w")
    wave, pot = initialize(res)

    for j = 1:100

        PE, KE = energy(wave, pot, 0.1, res)
 
        for i = 1:res
            wave[i] *= PE[i]
        end
 
        wave = fft(wave)

        for i = 1:res
            wave[i] *= KE[i]
        end

        wave = abs(ifft(wave))

        norm_const = 0.0
        for i = 1:res
            norm_const += wave[i]
        end

        println(norm_const)

        wave *= 1/norm_const

        #if j % 100 == 0
            for i = 1:res
                println(output, wave[i])
            end

            print(output, '\n', '\n')
        #end

    end
end

# Main

splitstep(1000)
println(hbar, '\t', scatl, '\t', boson_num, '\t', mass, '\t', coupling, '\t',
        radius, '\t', R, '\t', xmax)
