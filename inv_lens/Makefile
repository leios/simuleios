BIN = lens
CXX = g++
CXXFLAGS = -std=c++11 -g -Wall

plot: clean_dat $(BIN)
	./$(BIN) > /dev/null
	gnuplot plot.gp -

build: $(BIN)

run: clean_dat $(BIN)
	./$(BIN) 

$(BIN): optics_gustorn.cpp
	$(CXX) $(CXXFLAGS) optics_gustorn.cpp -o $(BIN) 

clean_dat:
	rm -Rf out.dat

clean: clean_dat
	rm -Rf $(BIN)
