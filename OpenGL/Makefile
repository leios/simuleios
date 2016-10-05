# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native

OGLFLAGS = -lGLEW -lglfw -lGL -lX11 -lpthread -lXrandr -lXi

BINS = simpleGL
OBJ = simpleGL.o
DEPS = simpleGL.h shader.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(OGLFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OGLFLAGS) -o $(BINS) $^
	./simpleGL
	#time ./simpleGL >> /dev/null

clean:
	rm -Rf $(BINS) #$(OBJ)

