# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2 -flto -Wno-sign-compare -fsanitize=address

CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = BEC_vis
OBJ = BEC_vis.o
DEPS = BEC_vis.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $^
	#./BEC_vis
	#convert -delay 5 -loop 0 /tmp/*.png frames/animation.gif

clean:
	rm -Rf $(BINS) $(OBJ) file*

