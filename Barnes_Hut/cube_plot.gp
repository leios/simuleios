#p octree data
set terminal x11 size 1250,1000

splot 'out.dat' u 1:2:3:(1) w l ls 1 lw 2
rep 'pout.dat' u 1:2:3:(1) ls 1 lw 2 lt rgb "red"
