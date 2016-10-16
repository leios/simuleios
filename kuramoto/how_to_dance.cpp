/*-------------how_to_dance.cpp-----------------------------------------------//
*
* Purpose: We are teaching Steve the Kuramoto oscillator how to dance with other
*          Kuramoto oscillators
*
*   Notes: This is a 1D simulation
*          compile with g++ how_to_dance.cpp
*          Note: Add in difference constants for difference dancers
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

/*----------------------------------------------------------------------------//
* STRUCTS / FUNCTIONS
*-----------------------------------------------------------------------------*/

// Struct to hold oscillator data
// Note: All oscillators in the group have the same phase / freq
struct oscillator{
    double pos, ppos, acc, phase, freq;

    oscillator() : pos(0.0), ppos(0.0), acc(0.0), phase(0.0), freq(0.0) {}
    oscillator(double p, double pp, double a, double ph, double f) :
               pos(p), ppos(pp), acc(a), phase(ph), freq(f) {}
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
void synchronize(std::vector<oscillator> &group, oscillator &steve, double dt,
                 double cutoff, std::ofstream &file);

// Find acceleration of all dancers on the dancefloor
void find_acc(std::vector<oscillator> &group, oscillator &steve, double cutoff);

// Function to calculate change in position
void verlet(std::vector<oscillator> &group, double dt);

// Function to output all positions
void output_pos(std::vector<oscillator> &group, oscillator &steve, 
                std::ofstream &file);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){

    // Initialize group
    std::vector<oscillator> group = init_group(20, 20.0, 20.0);

    // Initializing steve
    oscillator steve = init_steve(0.25 * M_PI, 20.0);

    std::ofstream file("particle_output.dat", std::ofstream::out);

    synchronize(group, steve, 0.1, 10, file);

    file.close();
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
        group[i].ppos = group[i].pos;
        //std::cout << group[i].pos << '\n';
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
void synchronize(std::vector<oscillator> &group, oscillator &steve, double dt,
                 double cutoff, std::ofstream &file){

    double sum;
    double nat_freq = steve.freq;

    // Now we need to synchronize steve to the group
    // Note: everyone is oscillating at the same phaseuency except for steve
    std::cout << steve.freq - group[0].freq << '\n';
    while ((steve.freq - group[0].freq) > 0.01){
        update_phase(group, steve, dt);

        // Note that all frequencies of members in the group are the same
        sum = sin(group[0].phase - steve.phase);
        steve.freq = nat_freq + sum;

        // Update positions for members in the group
        find_acc(group, steve, cutoff);
        verlet(group, dt);
        output_pos(group, steve, file);

        //std::cout << sum << '\t' << steve.freq - group[0].freq << '\n';
    }
}

// Function to calculate change in position
void verlet(std::vector<oscillator> &group, double dt){

    // changing the position of all dancers in simulation
    for (auto& member : group){
        std::cout << member.acc << '\n';
        double temp_x = member.pos;
        member.pos = 2 * member.pos - member.ppos + member.acc * dt*dt;
        member.ppos = temp_x;
        if (member.pos > 10){
            member.pos = 10;
        }
        if (member.pos < -10){
            member.pos = -10;
        }
    }
}

// Find acceleration of all dancers on the dancefloor
void find_acc(std::vector<oscillator> &group, oscillator &steve, double cutoff){

    double x_diff;

    // checking how far off steve is
    for (auto& member : group){
        x_diff = member.pos - steve.pos;
        // Repulsive force
        if ((steve.freq - member.freq) > cutoff){
            member.acc = 0.1 / (x_diff * x_diff);
        }
        // Attractive force
        else{
            member.acc = 0.1 * x_diff;
        }

        //std::cout << member.acc << '\n';
    }
}

// Function to output all positions
void output_pos(std::vector<oscillator> &group, oscillator &steve, 
                std::ofstream &file){

    std::vector<std::string> filenames(group.size());

    for (int i = 0; i < group.size(); i++){
        filenames[i] = "file" + std::to_string(i) + ".dat";
    }

    std::vector<std::ofstream> output_files;
    output_files.reserve(group.size());

    for (int i = 0; i < group.size(); i++){
        output_files.emplace_back(filenames[i], std::ofstream::app);
    }

    for (int i = 0; i < group.size(); i++){
        output_files[i] << group[i].pos << '\n';
    }

/*
    // loop for closing everything
    for (auto& file : output_files){
        file.close();
    }
*/

    // Output steve first
    file << steve.pos << '\t' << steve.phase << '\n';

    // Outputting all other people
    for (auto& member : group){
        file << member.pos << '\t' << member.phase << '\n';
    }

    file << '\n' << '\n';
}
