BINS = fdtd geometrical_optics FDTDTEz evanescent
CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer

.SUFFIXES: .cpp .dat .gp .plot

$(BINS): $@.cpp
        $(CXX) $(CXXFLAGS) $? -o $@

# When we do 'make foo.plot'
# 1) build foo if it's out of date compared foo.cpp
# 2) invoke foo
# 3) call gnuplot with foo.gp as input
# Will fail if invoked for foo.plot when there is no foo.gp
.gp.plot: $*
        ./$* > /dev/null
        gnuplot $*.gp -

# This is a special case for FDTD_para which compiles FDTD.cpp with OpenMP
# support, resulting in the binary FDTD_para
FDTD_para:
        $(CXX) $(CXXFLAGS) -fopenmp $@.cpp -o $@

.PHONY: clean *.plot
clean:
        rm -rf $(BINS)

