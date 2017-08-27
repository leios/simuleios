/*------------verlet.cpp------------------------------------------------------//
*
* Purpose: Simple demonstration of verlet algorithm
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>

// Simple function to use verlet integration
double verlet(double curr_p, double prev_p, double acc, double dt);

int main(){

    std::ofstream file("out.dat");

    double pos, curr_pos, prev_pos, time;
    pos = 1000;
    prev_pos = pos;
    time = 0.0;
    while (pos > 0){
        time += 0.01;
        curr_pos = pos;
        pos = verlet(curr_pos, prev_pos, -10, 0.01);
        prev_pos = curr_pos;

        file << pos << '\n';
    }

    std::cout << time << '\n';
    file.close();
}

// Simple function to use verlet integration
double verlet(double curr_p, double prev_p, double acc, double dt){
    double position;
    position = 2*curr_p - prev_p + acc * dt*dt;
    return position;
}
