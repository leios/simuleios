set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 1200, 800 
set output 'check.png'
unset key
set view map
set size ratio 1 1,1
set xtics border in scale 0,0 mirror norotate  autojustify
set ytics border in scale 0,0 mirror norotate  autojustify
set ztics border in scale 0,0 nomirror norotate  autojustify
set cbtics border in scale 0,0 mirror norotate  autojustify
set cbtics norangelimit 
set cbtics ("0" -3.14159, "2pi" 3.14159)
set title "Domain coloring of output" 
set xrange [ * : * ] noreverse writeback
set yrange [ * : * ] noreverse writeback
set cblabel "Phase Angle" 
set cblabel  offset character -2, 0, 0 font "" textcolor lt -1 rotate
set cbrange [ -3.14159 : 3.14159 ] noreverse nowriteback
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model HSV 
set palette defined ( 0 0 1 1, 1 1 1 1 )

thresh = 0.1
Hue(x,y) = (pi + atan2(-y,-x)) / (2*pi)
r(x,y) = sqrt(x*x + y*y)
theta(x,y) = atan2(y,x)
z(x,y) = r(x,y)*exp(theta(x,y)*sqrt(-1))
ip(x,y) = imag(z(x,y))
rp(x,y) = real(z(x,y))
val(x,y) = 0.5 + 0.5*(abs(z(x,y))-floor(abs(z(x,y))))
color(x,y) = hsv2rgb(Hue(rp(x,y), ip(x,y)), abs(z(x,y)), val(x,y)*shade(x,y))
shade(x,y) = (abs(sin(rp(x,y)*pi)**thresh) * abs(sin(ip(x,y)*pi))**thresh)
save_encoding = "utf8"

## Last datafile plotted: "++"
splot 'out.dat' using 1:2:(color($3,$4)) with pm3d lc rgb variable nocontour
