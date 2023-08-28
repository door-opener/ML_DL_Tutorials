#!usr/bin/gnuplot

set terminal pngcairo background rgb 'white' enhanced font "Times-New-Roman,20" fontscale 1.0 size 600, 400
set output 'Logistic_classification.png'

#set terminal x11

set style line 1  pointtype 9 pointsize default linecolor rgb "orange"
set style line 2  pointtype 13 pointsize default linecolor rgb "blue"
set style line 3  linecolor rgb "black" linewidth 2.5000 dashtype 1

set xrange [-2.0:2.0]
set yrange [-2.0:2.0]

set xtics -2.0, 1.0, 2.0 nomirror
set ytics -2.0, 1.0, 2.0 nomirror

set xlabel "Feature {/:Italic x_{1}}"
set ylabel "Feature {/:Italic x_{2}}"

set grid

set key top left nobox font "Times-New-Roman,8"

# Values of m and c calculated from logistic_regression_classifier.py #
f(x) = -2.18*x + 0.219

# Color of data points in txt file are labelled based on the value in column 3 of .txt #
plot "data_norm.out" u 1:2:3 w p title "Normalised data" pointtype 6 pointsize 1.5 lc variable \
, f(x) w l title "Decision Boundary"

#pause -1
