# Usage: make sim=<one of the values from BINS> to compile and plot
# the relevant simulation
#
# Do make sim=<value> compile if you only want to compile

BINS = fdtd fdtd_tez evanescent geometrical
CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2

compile: $(sim)

plot: $(sim)
	./$(sim) > /dev/null
	gnuplot plot_$(sim).gp -

%: %..cpp
	$(CXX) $(CXXFLAGS) $*.cpp -o $*

clean:
	rm -Rf $(BINS)

