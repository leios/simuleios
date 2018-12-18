do for [t=0:100] {
    plot for [ind=0:t] "barnsley.dat" i ind notitle pointtype 15 pointsize 0.2
    pause 0.1
}
