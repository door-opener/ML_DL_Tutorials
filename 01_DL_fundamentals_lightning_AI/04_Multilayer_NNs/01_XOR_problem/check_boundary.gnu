#!usr/bin/gnuplot

#set terminal pngcairo background rgb 'white' enhanced font "Times-New-Roman,20" fontscale 1.0 size 600, 400
#set output 'Logistic_classification.png'

set terminal x11

set style line 1  pointtype 9 pointsize default linecolor rgb "orange"
set style line 2  pointtype 13 pointsize default linecolor rgb "blue"
set style line 3  linecolor rgb "black" linewidth 2.5000 dashtype 1

set xrange [-5.0:5.0]
set yrange [-5.0:5.0]

set xtics -5.0, 1.0, 5.0 nomirror
set ytics -5.0, 1.0, 5.0 nomirror

set xlabel "Feature {/:Italic x_{1}}"
set ylabel "Feature {/:Italic x_{2}}"

set grid

set key top left nobox font "Times-New-Roman,8"

# Color of data points in txt file are labelled based on the value in column 3 of .txt #
plot "xor.txt" u 1:2:3 w p pointtype 6 pointsize 1.5 lc variable \

pause -1
