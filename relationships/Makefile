CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -O2 -fno-omit-frame-pointer -flto -Wno-sign-compare #-fsanitize=address

CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = relationship
OBJ = relationship.o cairo_vis.o
DEPS = ../visualization/cairo/cairo_vis.h

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $^
	#./relationship
	#convert -delay 5 -loop 0 /tmp/*.png frames/animation.gif

cairo_vis.o: ../visualization/cairo/cairo_vis.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

clean:
	rm -Rf $(BINS) $(OBJ) file*

