/*------------MD.cpp----------------------------------------------------------//
*
*             MD.cpp -- a simple event-driven MD simulation
*
* Purpose: Figure out the pressuer on the interior of a box based on 
*          Boltzmann's theory and stuff
*
*   Notes: This simulation will predominantly follow the logic of the link in
*          the README file
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <fstream>
#include <random>

using namespace std;
using simtime = int;

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct particle{
    int PID, ts;
    double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;
};

// holds interaction data
struct interact{
    int ts, part1, part2;
};

// Makes starting data
vector <particle> populate(int pnum, double box_length, double max_vel);

// Makes the list for our simulation later, required starting data
vector <interact> make_list(vector <vector <particle> > curr_data);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(vector<int> interactions, vector <particle> curr_data);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(void){

    return 0;
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Makes starting data
vector <particle> populate(int pnum, double box_length, double max_vel){
    vector <particle> curr_data;
    particle temp;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double>dis(0.0, 1.0);


    for (int i = 0; i < pnum; i++){

        temp.PID = i;
        temp.ts = 0;

        temp.pos_x = dis(gen) * box_length;
        temp.pos_y = dis(gen) * box_length;
        temp.pos_z = dis(gen) * box_length;
        temp.vel_x = dis(gen) * max_vel;
        temp.vel_y = dis(gen) * max_vel;
        temp.vel_z = dis(gen) * max_vel;

        curr_data.push_back(temp);
    
    }

    return curr_data;
}

// Makes the list for our simulation later, required starting data
// Step 1: Check interactions between particles, based on README link
// Step 2: Update list.
vector <interact> make_list(vector <particle> curr_data){
    vector <interact> list;

    return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
void simulate(vector<int> interactions, vector <particle> curr_data){


}

