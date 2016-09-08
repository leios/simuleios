#p octree data
set terminal x11 size 1250,1000

splot 'octree0.dat' u 1:2:3:(1) w l ls 1 lw 2
rep 'pout.dat' u 1:2:3:(1) i 0 ls 1 lw 2 lt rgb "red"
