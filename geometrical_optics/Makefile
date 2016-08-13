# Usage: 'make sim=<one of the values from BINS>' to compile and plot
# the relevant simulation
#
# Do 'make sim=<value> compile' if you only want to compile
# Or 'make sim=<value> run' if you only want to run and not plot

BINS = fdtd fdtd_tez geometrical
CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2
sim = geometrical

plot: $(sim)
	./$(sim) > /dev/null
	gnuplot $(sim).gp -

compile: $(sim)

run: $(sim)
	./$(sim)

%: %..cpp
	$(CXX) $(CXXFLAGS) $*.cpp -o $*

clean:
	cp $(sim).dat backup.dat
	rm -Rf $(BINS)

