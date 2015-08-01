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

using namespace std;

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct data{
    double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;
};

// Makes starting data
vector <data> populate();

// Makes the list for our simulation later, required starting data
vector <int> make_list(vector <data> curr_data);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(vector<int> interactions, vector <data> curr_data);

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
vector <data> populate(){
    vector <data> curr_data;

    return curr_data;
}

// Makes the list for our simulation later, required starting data
vector <int> make_list(vector <data> curr_data){
    vector <int> list;

    return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(vector<int> interactions, vector <data> curr_data){

}

