using Images, ImageMagick
function fourier_op_out(N::Int64, xmax::Float64, output_file::String)
    n = [2*xmax * i / N for i = 1:N]
    k = n'
    #n = [-xmax + 2*xmax*i/N for i =0:N-1]
    #k = append!(n[1:div(N,2)],n[div(N,2):N-1])'
    transform_matrix = (exp.(-2im*pi*n*k/N).+(1.0+1im))/2
    a = Array{RGB{N0f8},2}(undef, N, N)
    for i = 1:N
        for j = 1:N
            a[i,j] = RGB{N0f8}(real(transform_matrix[i,j]), 0, imag(transform_matrix[i,j]))
        end
    end

    save(output_file, a)

end

function DFT_out(N::Int64, output_file::String)
    n = [i for i = 0:N-1]
    k = n'
    transform_matrix = (exp.(-2im*pi*n*k/N).+(1.0+1im))/2
    a = Array{RGB{N0f8},2}(undef, N, N)
    for i = 1:N
        for j = 1:N
            a[i,j] = RGB{N0f8}(real(transform_matrix[i,j]), 0, imag(transform_matrix[i,j]))
        end
    end

    save(output_file, a)

end

