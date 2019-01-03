function DFT_out(N::Int64, output_file::String)
    n = 0:N-1
    k = n'
    transform_matrix = exp.(-2im*pi*n*k/N)
    a = Array{RGB{N0f8},2}(undef, N, N)
    for i = 1:N
        for j = 1:N
            #a[i,j] = RGB{N0f8}(abs(real(transform_matrix[i,j])), 0, abs(imag(transform_matrix[i,j])))
            a[i,j] = RGB{N0f8}(0, abs2(transform_matrix[i,j]), 0)
        end
    end

    save(output_file, a)

end
