#!usr/bin/gnuplot

set terminal x11

set style line 1  pointtype 9 pointsize default linecolor rgb "orange"
set style line 2  pointtype 13 pointsize default linecolor rgb "blue"
set style line 3  linecolor rgb "black" linewidth 2.5000 dashtype 1

set xrange [-4.0:4.0]
set yrange [-4.0:4.0]

set xlabel "Feature x_{1}"
set ylabel "Feature x_{2}"

set key top left nobox

# Values of m and c calculated from perceptron.py #
f(x) = -1.1122222222222222*x + 1.5151515151515151

# Color of data points in txt file are labelled based on the value in column 3 of .txt #
plot "perceptron_toydata-truncated.txt" u 1:2:3 w p title "Raw data" pointtype 6 pointsize 1.5 lc variable \
, f(x)  title "Decision Boundary"


pause -1
