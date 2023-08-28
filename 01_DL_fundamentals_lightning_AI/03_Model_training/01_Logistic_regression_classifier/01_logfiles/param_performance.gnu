#!usr/bin/gnuplot

set terminal pngcairo background rgb 'white' enhanced font "Times-New-Roman,20" fontscale 1.0 size 600, 400
set output 'Loss_vs_Epoch.png'

set style line 1  pointtype 9 pointsize default linecolor rgb "orange"
set style line 2  pointtype 13 pointsize default linecolor rgb "blue"
set style line 3  linecolor rgb "black" linewidth 2.5000 dashtype 1

set xrange [0.0:21.0]
set yrange [0.0:1.0]
set ytics nomirror
set xtics nomirror
#set yrange [-4.0:4.0]

set xlabel "Epoch Number"
set ylabel "Loss"

set key top right nobox font "Times-New-Roman,10"

plot "log_10_0.05.out" u 1:2 w lp title "Batchsize = 10, Learning rate = 0.05" \
, "log_10_0.1.out" u 1:2 w lp title "Batchsize = 10, Learning rate = 0.1" \
, "log_10_0.2.out" u 1:2 w lp title "Batchsize = 10, Learning rate = 0.2" \
, "log_5_0.05.out" u 1:2 w lp title "Batchsize = 5, Learning rate = 0.05" \
, "log_5_0.1.out" u 1:2  w lp title "Batchsize = 5, Learning rate = 0.1" \
, "log_5_0.2.out" u 1:2 w lp title "Batchsize = 5, Learning rate = 0.2"

