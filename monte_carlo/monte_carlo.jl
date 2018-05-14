# function to integrate a unit circle to find pi via monte_carlo
function monte_carlo(n::Int64)

    pi_count = 0
    for i = 1:n
        point_x = rand()
        point_y = rand()

        if (point_x^2 + point_y^2 < 1)
            pi_count += 1
        end
    end

    pi_estimate = 4*pi_count/n
    println(pi - pi_estimate)
end

monte_carlo(10000000)
