CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2 -flto

BINS = barnes_hut
OBJ = barnes_hut.o
DEPS = vec.h 

%.o : %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(BINS) $^
	./barnes_hut

clean:
	rm -Rf $(BINS) $(OBJ)
