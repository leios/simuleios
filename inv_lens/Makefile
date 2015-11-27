CXX = clang++
CXXFLAGS = -std=c++11 -g -Wall -O2
BINS= FTDT geometrical_optics FDTDTEz evanescent

plot: clean $(BIN)
        ./$(BIN) > /dev/null
        gnuplot $@.gp -

$(BINS): $@.cpp
        $(CXX) $(CXXFLAGS) $@.cpp -o $@

FTDT_para: FTDT.cpp
        $(CXX) $(CXXFLAGS) -fopenmp $@.cpp -o $@

.PHONY: clean
clean:
        rm -rf *.dat $(BINS)
