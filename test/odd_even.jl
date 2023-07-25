function check(n1, n2, n3, n4)
    all_states = zeros(Bool, n1+1,n2+1,n3+1,n4+1)

    for i = 0:n1
        for j = 0:n2
            for k = 0:n2
                for l = 0:n2
                    all_states[i+1,j+1,k+1,l+1] = isodd(i+j+k+l)
                end
            end
        end
    end

    return all_states
end

function check2(n1, n2, n3, n4)
    all_states = zeros(Bool, n1+1,n2+1,n3+1,n4+1)

    for i = 0:2:(n1+1)*(n2+1)*(n3+1)*(n4+1)-2
        println(i)
        v1 = Int(i%(n1+1))
        v2 = Int((i%((n1+1)*(n2+1))-v1)/(n1+1))
        v3 = Int(((i%((n1+1)*(n2+1)*(n3+1))-v1)/(n1+1)-v2)/(n2+1))
        v4 = Int(((((i-v1)/(n1+1))-v2)/(n2+1)-v3)/(n3+1))
        all_states[v1+1, v2+1, v3+1, v4+1] = true
    end

    return all_states

end
