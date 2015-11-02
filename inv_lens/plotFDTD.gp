#p "FDTD.dat"
set dgrid3d 100, 100, 1
set pm3d
set palette
set palette color
set pm3d map
set palette defined ( 0 "green", 1 "blue", 2 "red", 3 "orange" )
splot "FDTD.dat" i 10 u 2:3:4
