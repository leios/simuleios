#p "FDTD.dat"
set terminal x11 size 1250,1000
set dgrid3d 200, 200, 2
set pm3d
set palette
set palette color
set pm3d map
set palette defined ( 0 "green", 1 "blue", 2 "red", 3 "orange" )
splot "FDTD.dat" i 100 u 2:3:4 

# PS: If you want to watch everything run with time, uncomment the following:
# set cbrange [-0.1:0.1]
# do for [ii=1:101:1] { splot "FDTD.dat" i ii u 2:3:4 }
