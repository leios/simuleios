CXX = g++
CXXFLAGS = -std=c++14 -g -Wall -march=native -fopenmp -O2 -fno-omit-frame-pointer -flto -Wno-sign-compare #-fsanitize=address

CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = definitely_a_date
OBJ = definitely_a_date.o cairo_vis.o
DEPS = ../visualization/cairo/cairo_vis.h

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $^

cairo_vis.o: ../visualization/cairo/cairo_vis.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

clean:
	rm -Rf $(BINS) $(OBJ) file*

