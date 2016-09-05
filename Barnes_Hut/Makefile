# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native

OGLFLAGS = -lGLEW -lglfw -lGL -lX11 -lpthread -lXrandr -lXi

BINS = barnes_hut
OBJ = barnes_hut.o
DEPS = vec.h barnes_hut.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(OGLFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OGLFLAGS) -o $(BINS) $^
	./barnes_hut >> particle_check.dat

clean:
	rm -Rf $(BINS) #$(OBJ)

