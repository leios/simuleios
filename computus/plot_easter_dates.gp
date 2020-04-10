set terminal epslatex standalone
set output "check.tex"

set xlabel "Year"

set ylabel "Date"
set yrange [0:35]

set xtics ("2020" 0, "2045" 25, "2070" 50, "2095" 75, "2119" 99)
set ytics ("March 22nd" 0, "March 31st" 9, "April 9th" 18, "April 18th" 27, "April 26th" 35)

p "dates.dat" w l lw 2 title ""
