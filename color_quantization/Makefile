# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -fopenmp #-g -Wall -march=native -fsanitize=address

OGLFLAGS = -lGLEW -lglfw -lGL -lX11 -lpthread -lXrandr -lXi

BINS = color_quantization
OBJ = color_quantization.o
DEPS = vec.h coloir_quantization.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(OGLFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OGLFLAGS) -o $(BINS) $^
	#./color_quantization
	time ./color_quantization >> /dev/null

clean:
	rm -Rf $(BINS) $(OBJ) *.dat

