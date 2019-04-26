set terminal png
set view map
set size square
res = 512
set xrange[1:res]
set yrange[1:res]
set palette model CMY rgbformulae 7,5,15
list = system('ls data/*.dat')
i = 0
do for [file in list] {
    set output sprintf("images/fractal_zoom%05d.png", i)
    splot file matrix with image
    print(i)
    i = i + 1
}
