# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2
CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = geometrical
OBJ = geometrical.o optics_vis.o
DEPS = geometrical.h optics_vis.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $^
	./geometrical
	#convert -delay 5 -loop 0 frames/*.png frames/animation.gif

clean:
	rm -Rf $(BINS) $(OBJ)

