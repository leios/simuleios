/*-------------how_to_dance.cpp-----------------------------------------------//
*
* Purpose: We are teaching Steve the Kuramoto oscillator how to dance with other
*          Kuramoto oscillators
*
*   Notes: This is a 1D simulation
*          compile with g++ how_to_dance.cpp
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <vector>
#include <random>

/*----------------------------------------------------------------------------//
* STRUCTS / FUNCTIONS
*-----------------------------------------------------------------------------*/

// Struct to hold oscillator data
// Note: All oscillators in the group have the same phase / freq
struct oscillator{
    double pos, vel, acc, phase, freq;

    oscillator() : pos(0.0), vel(0.0), acc(0.0), freq(0.0), phase(0.0) {}
    oscillator(double p, double v, double a, double ph, double f) :
               pos(p), vel(v), acc(a), phase(ph), freq(f) {}
};

// Function to initialize group, dance floor is the size of the box(?)
std::vector<oscillator> init_group(int groupnum, double freq, 
                                   double dance_floor);

// Function for initializing steve
// Notes: We are assuming steve waltzes to the center of the dance_floor
//        We can change the position later if we want, and we can generate
//            a random freq too.
oscillator init_steve(double phase, double max_freq);

// Function to update phaseuency of steve and group
void update_phase(std::vector<oscillator> &group, oscillator &steve,
                  double dt);

// Function to synchronize steve to the rest of the group
// Note: This means the rest of the group is oscillating at the same phase
void synchronize(std::vector<oscillator> &group, oscillator &steve, double dt);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){

    // Initialize group
    std::vector<oscillator> group = init_group(20, 20.0, 20.0);

    // Initializing steve
    oscillator steve = init_steve(0.25 * M_PI, 20.0);

    synchronize(group, steve, 0.1);
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function to initialize group
std::vector<oscillator> init_group(int groupnum, double freq,
                                   double dance_floor){
    std::vector<oscillator> group(groupnum);

    // Define random distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        dance_floor_dist(-dance_floor * 0.5, dance_floor * 0.5);

    for (int i = 0; i < groupnum; i++){
        group[i] = oscillator(dance_floor_dist(gen), 0.0, 0.0, freq, 0.0);
    }

    return group;

}

// Function for initializing steve
oscillator init_steve(double phase, double max_freq){

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        steve_dist(0, 2 * M_PI);
    oscillator steve = oscillator(0.0, 0.0, 0.0, steve_dist(gen), phase);
    return steve;

}

// Function to update phaseuency of steve and group
void update_phase(std::vector<oscillator> &group, oscillator &steve,
                      double dt){
    // group first
    for (size_t i = 0; i < group.size(); i++){
        group[i].phase += group[i].freq * dt;
    }

    // now for ... steve
    steve.phase += steve.freq * dt;
}

// Function to synchronize steve to the rest of the group
// Note: we can weight the dancers based on their proximity to steve
void synchronize(std::vector<oscillator> &group, oscillator &steve, double dt){

    double sum;
    double nat_freq = steve.freq;

    // Now we need to synchronize steve to the group
    // Note: everyone is oscillating at the same phaseuency except for steve
    std::cout << steve.freq - group[0].freq << '\n';
    while ((steve.freq - group[0].freq) > 0.01){
        update_phase(group, steve, dt);
        //update_pos(group, dt);
        sum = sin(group[0].phase - steve.phase);
        steve.freq = nat_freq + sum;

        std::cout << sum << '\t' << steve.freq - group[0].freq << '\n';
    }
}
