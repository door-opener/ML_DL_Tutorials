#!usr/bin/gnuplot

set terminal pngcairo background rgb 'white' enhanced font "Times-New-Roman,16" fontscale 1.0 size 600, 400
set output 'Loss_vs_iteration_mnist.png'

#set terminal x11

set style line 1  pointtype 2 pointsize 0.5 linecolor rgb "purple"
set style line 2  pointtype 13 pointsize default linecolor rgb "blue"
set style line 3  linecolor rgb "yellow" linewidth 2.5000 dashtype 1

set xrange [0.0:9000.0]
set yrange [0.0:2.5]
set xtics 0.0, 1000, 9000.0 nomirror
set ytics nomirror

set xlabel "Iteration #"
set ylabel "Loss"

set grid

set key top right nobox font "Times-New-Roman,14"

# Color of data points in txt file are labelled based on the value in column 3 of .txt #
plot "training_loss.dat" u 1:2 w p ls 1 title "Training Loss", "averaged_loss.dat" u 1:2 w l ls 3 title "Averaged Loss" 

#pause -1
