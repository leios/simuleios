#first target is used if no target is given to make
compile:
	g++ -std=c++14 -Wall -Wextra MD.cpp -o MD #lets not do pedantic

debug:
	g++ -g gdb -std=c++11 MD.cpp -o MD
	gdb ./MD

plot:
	gnuplot "out.dat"  u 2:5 i 0 w 1  # is that right?
clean:
	rm MD
	#should clean remove out.dat as well? 
	# rm out.dat
run: compile
	./MD
upload:
	git commit
	git push
dosomethingcool:
	cowsay "I have no idea what I am doing."

