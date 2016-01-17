set terminal x11 size 1250,1000
set xrange [-2:2]
set yrange [-2:2]
set zrange [-2:2]

do for [ii=0:59:1] { splot "out.dat" i ii u 1:2:3, "out.dat"i ii u 4:5:6; pause .5;} 
