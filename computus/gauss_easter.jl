function computus(year)

    # Tracking lunar orbiting
    # Year's position on the 19 year lunar phase cycle
    a = mod(year, 19)
    
    # Century index
    k = floor(year/100)

    # shift of metonic cycle
    p = floor((13 + 8 * k) / 25) 

    # Leap-day difference between Julian and Gregorian calendars?
    q = floor(k / 4)

    # Correction to starting point of calculation each century
    M = mod(15 - p + k - q, 30)

    # Finding the next Sunday
    # The difference in the number of leap days between the Gregorian and
    # Julian calendars
    N = mod(4 + k - q, 7)

    # Number of days from March 21st until the full moon
    d = mod(19 * a + M, 30)

    # 52x7 = 364, but there are 365 days a year, so we are offsetting for this.
    # 2*b + 4*c  will drop by 1 every year, but 2 on leap years
    b = mod(year, 4)
    c = mod(year, 7)

    # Ignoring the 6*d term, this is the number of days until the next sunday
    # from March 22. The 6*d adds the appropriate offset to the next Sunday.
    e = mod(2 * b + 4 * c + 6 * d + N, 7)

    # uncomment to recreate Servois's table
    # return mod(21 + d, 31)
    if(22+ d + e > 31)
        return string(d + e - 9)*" April"
    else
        return string(22+d+e)*" March"
    end
end

easter = computus(1800)
