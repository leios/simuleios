set terminal epslatex standalone color
set output "servois.tex"

set size square
set view map

set xtics 0,1,10
set xtics offset 0, 17

#set xtics format ""

set xrange [-0.5:9.5]
set yrange [9.5:-0.5]
set cbrange [1:31]

set title ""

#set label "10" at 0,9 center front
set xtic scale 0
set ytic scale 0

set ylabel "Decade"
set xlabel "Year"
set xlabel offset 0, 20
set cblabel "Day"

#set ytics ( \
#    "200" 0,\
#    "201" 1,\
#    "202" 2,\
#    "203" 3,\
#    "204" 4,\
#    "205" 5,\
#    "206" 6,\
#    "207" 7,\
#    "208" 8,\
#    "209" 9 \
#)
set ytics ( \
    "180" 0,\
    "181" 1,\
    "182" 2,\
    "183" 3,\
    "184" 4,\
    "185" 5,\
    "186" 6,\
    "187" 7,\
    "188" 8,\
    "189" 9 \
)


# palette
#set palette defined ( 0 '#2166AC',\
#                      1 '#4393C3',\
#                      2 '#92C5DE',\
#                      3 '#D1E5F0',\
#                      4 '#FDE0EF',\
#                      5 '#F1B6DA',\
#                      6 '#DE77AE',\
#                      7 '#C51B7D')
set palette defined ( 0 '#4393C3',\
                      1 '#92C5DE',\
                      2 '#D1E5F0',\
                      3 '#FDE0EF',\
                      4 '#F1B6DA',\
                      5 '#DE77AE')

#splot "servois\_2000" matrix with image title "", \
#      "servois\_2000" using 1:2:0:(sprintf("%d",$3)) matrix with labels notitle
splot "servois\_1800" matrix with image title "", \
      "servois\_1800" using 1:2:0:(sprintf("%d",$3)) matrix with labels notitle
