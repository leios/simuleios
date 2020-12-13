using FFTW

function split_and_zeropad(a, b)
    a_digits = digits(a)
    b_digits = digits(b)

    return vcat(a_digits, zeros(Int, length(b_digits))),
           vcat(b_digits, zeros(Int, length(a_digits)))
end

# TODO: there is probably a better way to do the carrying operations
#       check biginteger source code.
function vector_to_int(v)
    sum = 0
    for i = 1:length(v)
        sum += 10^(i-1)*v[i]
    end

    return sum
end

function carry!(v::Vector{Int})
    for i = 1:length(v)-1
        digs = digits(v[i])
        v[i+1] += vector_to_int(digs[2:end])
        v[i] = digs[1]
    end

    return v
end

function schonhage_strassen(a, b)
    va, vb = split_and_zeropad(a, b)

    vf = round.(Int, real(ifft(fft(va).*fft(vb))))

    carry!(vf)

    return vector_to_int(vf)
end
