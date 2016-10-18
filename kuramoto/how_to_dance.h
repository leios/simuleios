/*-------------how_to_dance.h-------------------------------------------------//
*
* Purpose: This file is meant to visualize the kuramoto dancing simulator
*
*   Notes: This will be using Cairo
*
*-----------------------------------------------------------------------------*/

#ifndef HOW_TO_DANCE
#define HOW_TO_DANCE

#include "how_to_dance_vis.h"
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
    double pos, ppos, acc, phase, freq, vel, attr;

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
void synchronize(frame &anim, std::vector<oscillator> &group, oscillator &steve,
                 double dt, double cutoff, std::ofstream &file);

// Find acceleration of all dancers on the dancefloor
void find_acc(std::vector<oscillator> &group, oscillator &steve, double cutoff);

// Function to calculate change in position
void verlet(std::vector<oscillator> &group, double dt);

// Function to output all positions
void output_pos(std::vector<oscillator> &group, oscillator &steve, 
                std::ofstream &file);

// Function to return sign of double
double sign(double variable);

#endif
