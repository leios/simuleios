BINS = fdtd geometrical_optics FDTDTEz evanescent
CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer

plot: clean_dat $(BIN)
	./$(BIN) > /dev/null
	gnuplot plotFDTD.gp -

build: $(BIN)

run: clean_dat $(BIN)
	./$(BIN) 

$(BIN): $@.cpp
	$(CXX) $(CXXFLAGS) $@.cpp -o $@

clean_dat:
	rm -Rf $@.dat

clean: clean_dat
	rm -Rf $(BINS)
