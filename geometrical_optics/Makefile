# Makefile for huffman simulation

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2 -flto
#CXXFLAGS = -std=c++11 -O3 -s -fopenmp -pipe -flto -fmodulo-sched -fmodulo-sched-allow-regmoves -fgcse-sm -fgcse-las -fgcse-after-reload -funsafe-loop-optimizations -fipa-pta -ftree-loop-linear -floop-interchange -floop-strip-mine -floop-block -fgraphite-identity -floop-parallelize-all -ftree-loop-distribution -ftree-loop-im -ftree-loop-ivcanon -fivopts -ftracer -fvariable-expansion-in-unroller -freorder-blocks-and-partition -fweb -ffast-math -frename-registers -funswitch-loops -fvisibility=hidden -fvisibility-inlines-hidden

CAIROFLAGS = `pkg-config --cflags --libs cairo`
BINS = geometrical
OBJ = geometrical.o optics_vis.o
DEPS = geometrical.h optics_vis.h

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -c -o $@ $<

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) $(CAIROFLAGS) -o $(BINS) $^
	./geometrical
	#convert -delay 5 -loop 0 /tmp/*.png frames/animation.gif

clean:
	rm -Rf $(BINS) $(OBJ)

