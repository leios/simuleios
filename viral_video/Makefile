# Makefile for huffman simulation

CXX = g++
#CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2 -flto -Wno-sign-compare -fsanitize=address
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -O2 -fno-omit-frame-pointer -flto -Wno-sign-compare

CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = viral_video
OBJ = viral_video.o cairo_vis.o
DEPS = viral_video.h ../visualization/cairo/cairo_vis.h

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $^
	#./viral_video
	#convert -delay 5 -loop 0 /tmp/*.png frames/animation.gif

cairo_vis.o: ../visualization/cairo/cairo_vis.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

clean:
	rm -Rf $(BINS) $(OBJ) file*

