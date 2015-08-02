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

namespace sim{
	/* putting our stuff in its own namespace so we can call "time" "time"
	   without colliding with names from the standard library */
	using time = int;
}

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct Particle{
	int PID;
	sim::time ts;
	double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;
};

// holds interaction data
struct Interaction{
	sim::time ts;
	int part1, part2;
};

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel);

// Makes the list for our simulation later, required starting data
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(std::vector<int> interactions, std::vector<Particle> curr_data);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
	//no void and no return 0 for main
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel){
	std::vector<Particle> curr_data(pnum); //<- constructor instead of pushing

	static std::random_device rd; //static to create only 1 random_device
	static std::mt19937 gen(rd());
	/* instead of doing % and * to get the correct distribution we directly
	   specify which distribution we want */
	std::uniform_real_distribution<double> box_length_distribution(0, box_length);
	std::uniform_real_distribution<double> max_vel_distribution(0, max_vel);

	int PID_counter = 0;
	for (auto &p : curr_data){ //read: for all particles p in curr_data
		p.PID = PID_counter++;
		p.ts = 0;

		/* maybe you like this version more than the loop below
		p.pos_x = box_length_distribution(gen);
		p.pos_y = box_length_distribution(gen);
		p.pos_z = box_length_distribution(gen);
		p.vel_x = max_vel_distribution(gen);
		p.vel_y = max_vel_distribution(gen);
		p.vel_z = max_vel_distribution(gen);
		*/
		for (auto &pos : { &p.pos_x, &p.pos_y, &p.pos_z })
			*pos = box_length_distribution(gen);
		for (auto &vel : { &p.vel_x, &p.vel_y, &p.vel_z })
			*vel = max_vel_distribution(gen);
	}

	return curr_data;
}

// Makes the list for our simulation later, required starting data
// Step 1: Check interactions between Particles, based on README link
// Step 2: Update list.
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data){
	/* passing curr_data as const reference to avoid the copy and accidental
	   overwriting */
	std::vector<Interaction> list;

	return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
void simulate(std::vector<int> interactions, std::vector<Particle> curr_data){


}

