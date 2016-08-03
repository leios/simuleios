# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2
CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = huffman_vis
OBJ = huffman.o huffman_vis.o
#BINS = vitter
#OBJ = huffman.o vitter.o
DEPS = huffman.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $(BINS).cpp $<
	#./vitter
	./huffman_vis
	convert -delay 5 -loop 0 frames/*.png frames/animation.gif

clean:
	rm -Rf $(BINS) $(OBJ)

