# setting output to file of size 800 x 800
set terminal pngcairo size 800, 800 
set output 'out.png'

# removes legend
unset key

# projects image onto 2D plane
set view map

# sets aspect ratio of plot to be square 
set size square

# scaling the x, y, and colorbar tics to zero so they are not seen in the plot
set xtics border scale 0,0
set ytics border scale 0,0
set cbtics border scale 0,0

# setting tics in color bar at 0 and 2pi
set cbtics ("0" -3.14159, "2pi" 3.14159)
set title "Domain coloring of output" 

set cblabel "Phase Angle" 
set cbrange [ -3.14159 : 3.14159 ]

# use hsv for colorbar and set palette to use full hsv space
set palette model HSV
set palette defined ( 0 0 1 1, 1 1 1 1 )

# setting functions necessary for domain coloring
# setting threshold for gridlines. Smaller threashold will make smaller lines
thresh = 0.1
Hue(x,y) = (pi + atan2(-y,-x)) / (2*pi)
r(x,y) = sqrt(x*x + y*y)
theta(x,y) = atan2(x,y)
z(x,y) = r(x,y)*exp(theta(x,y)*sqrt(-1))
ip(x,y) = imag(z(x,y))
rp(x,y) = real(z(x,y))
val(x,y) = 0.5 + 0.5*(abs(z(x,y))-floor(abs(z(x,y))))
color(x,y) = hsv2rgb(Hue(rp(x,y), ip(x,y)), abs(z(x,y)), val(x,y)*shade(x,y))
shade(x,y) = (abs(sin(rp(x,y)*pi)**thresh) * abs(sin(ip(x,y)*pi))**thresh)
save_encoding = "utf8"

## Last datafile plotted: "++"
splot 'out.dat' using 1:2:(color($3,$4)) with pm3d lc rgb variable nocontour
