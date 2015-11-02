#p "FDTD.dat"
set dgrid3d 200, 200, 1
set pm3d
set palette
set palette color
set pm3d map
splot "FDTD.dat" i 199 u 2:3:4

