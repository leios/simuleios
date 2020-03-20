function computus(year; cal_type="gregorian")

    M = 0
    N = 0
    if (cal_type == "gregorian")
        # Tracking lunar orbiting
        a = mod(year, 19)
        b = mod(year, 4)
        c = mod(year, 7)

        k = floor(year/100)
        p = floor((13 + 8 * k) / 25) 

        q = floor(k / 4)

        M = mod(15 - p + k - q, 30)

        # Finding the next Sunday
        N = mod(4 + k - q, 7)
        d = mod(19 * a + M, 30)
        e = mod(2 * b + 4 * c + 6 * d + N, 7)
        # uncomment to recreate Servois's table
        # return mod(21 + d, 31)
        if(22+ d + e > 31)
            return string(d + e - 9)*" April"
        else
            return string(22+d+e)*" March"
        end
    elseif (cal_type == "julian")
        a = mod(year, 19)
        b = mod(year, 4)
        c = mod(year, 7)
        M = 15

        # Finding the next Sunday
        N = 6
        d = mod(19 * a + M, 30)
        e = mod(2 * b + 4 * c + 6 * d + N, 7)
        return (22+ d + e, d + e - 9)
    else
        println("Unknown calendar type ", cal_type, "!")
    end

end

easter = computus(1800; cal_type="gregorian")
