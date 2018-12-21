set xrange[-1.5:1.5]
set obj 1 rectangle behind from screen 0,0 to screen 1,1
set obj 1 fillstyle solid 1.0 fillcolor rgbcolor "black"
do for [t=0:100] {
    plot for [ind=0:t] "barnsley_tree.dat" i ind notitle pointtype 15 pointsize 0.2
    pause 0.1
}
