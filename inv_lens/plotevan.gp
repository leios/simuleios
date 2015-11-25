#p "FDTD.dat"
set terminal x11 size 1250,1000
set dgrid3d 200, 200, 2
set pm3d
set palette
set palette color
set pm3d map
set palette defined ( 0 "blue", 1 "white", 2 "red" )

# PS: If you want to watch everything run with time, uncomment the following:
# set cbrange [-0.05:0.05]

set cbrange [-0.2:0.2]
# do for [ii=1:99:1] { plot "evanescent.dat" i ii u 2:3:4 w image; pause .1}
plot "evanescent.dat" i 19 u 2:3:4 w image

set object circle at 100,100 size 50 fs empty border 30 lw 30
set size ratio -1
