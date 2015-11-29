#p "fdtd_tez.dat"
set terminal x11 size 1250,1000
set dgrid3d 200, 200, 2
set pm3d
set palette
set palette color
set pm3d map
# set palette defined ( 0 "green", 1 "blue", 2 "red", 3 "orange" )
# set palette defined ( 0 "black", 1 "white" )
set palette defined ( 0 "blue", 1 "white", 2 "red" )
# splot "fdtd_tez.dat" i 100 u 2:3:4
# set zrange [-1:1]

# PS: If you want to watch everything run with time, uncomment the following:
# set cbrange [-0.05:0.05]
set object circle at 100,100 size 80 fs empty border 1 front lw 5
set size ratio -1

set cbrange [-0.02:0.02]
# splot "fdtd_tez.dat" i 199 u 2:3:4
do for [ii=1:140:1] { plot "fdtd_tez.dat" i ii u 2:3:5 w image; pause .1}
# do for [ii=1:151:1] { splot "fdtd_tez.dat" i ii u 2:3:4}

set object circle at 100,100 size 50 fs empty border 30 lw 30
set size ratio -1
# rep
