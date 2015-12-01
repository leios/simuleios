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

#set object rect from 10, 65 to 290, 115 fs empty border 1 front lw 3
#set size ratio -1

set object rect from 10, 125 to 290, 175 fs empty border 1 front lw 3
set size ratio -1

#set object rect from 10, 185 to 290, 235 fs empty border 1 front lw 3
#set size ratio -1

set cbrange [-0.2:0.2]
do for [ii=1:80:1] { plot "evanescent.dat" i ii u 2:3:4 w image; pause .01}
# plot "evanescent.dat" i 19 u 2:3:4 w image
