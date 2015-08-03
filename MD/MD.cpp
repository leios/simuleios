/*------------MD.cpp----------------------------------------------------------//
*
*             MD.cpp -- a simple event-driven MD simulation
*
* Purpose: Figure out the pressure on the interior of a box based on
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
#include <algorithm> // std::sort, etc..

namespace sim{
	/* putting our stuff in its own namespace so we can call "time" "time"
	   without colliding with names from the standard library */
	using time = double;
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
	sim::time rtime;
	int part1, part2;
};

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel);

// Makes the list for our simulation later, required starting data
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, double radius);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(std::vector<int> interactions, std::vector<Particle> curr_data);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(void){

    int pnum = 100;
    double box_length = 10, max_vel = 0.01;
		double r = 0.1; // i have no clue what would be a good particle radius...

    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel);
		std::vector<Interaction> inter_data = make_list(curr_data, box_length, r);

		// i tried to iterate through the sorted list
		// All rtimes were NaN (not a number)
		// I don't know much about physics so I'll leave it like that
		// Sorting should work just like that though... Tested it with part1 (int)
		// I tried without sorting, still all were NaN, so it's not that.
    for (auto &i : inter_data){
			//std::cout << i.part1 << std::endl;

			std::cout << " x: " << curr_data[i.part1].pos_x <<
									 " y: " << curr_data[i.part1].pos_y <<
									 " z: " << curr_data[i.part1].pos_z << std::endl;

    }

    return 0;
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel){
    std::vector<Particle> curr_data(pnum);

    // static to create only 1 random_device
    static std::random_device rd;
    static std::mt19937 gen(rd());

    /* instead of doing % and * to get the correct distribution we directly
       specify which distribution we want */

    std::uniform_real_distribution<double>
        box_length_distribution(0, box_length);
    std::uniform_real_distribution<double>
       max_vel_distribution(0, max_vel);

    int PID_counter = 0;
    for (auto &p : curr_data){ //read: for all particles p in curr_data
        p.ts = 0;
        p.PID = PID_counter++;

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
// Step 3: Sort list, lowest first
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, double radius){
    /* passing curr_data as const reference to avoid the copy and accidental
    overwriting */
    std::vector<Interaction> list;
    Interaction test;
    int i = 0,j = 0;

    // Step 1 -- find interactions
    for (auto &ip : curr_data){
        for (auto &jp : curr_data){
            double del_x = ip.pos_x - jp.pos_x;
            double del_y = ip.pos_y - jp.pos_y;
            double del_z = ip.pos_z - jp.pos_y;

            double del_vx = ip.vel_x - jp.vel_y;
            double del_vy = ip.vel_y - jp.vel_y;
            double del_vz = ip.vel_z - jp.vel_z;

            double r_prime = 2 * radius;

            double rad_d = (pow(del_vx * del_x + del_vy * del_y
                          		+ del_vz * del_z, 2)
                    	 - 4 * (del_vx * del_vx + del_vy * del_vy
											 				+ del_vz * del_vz)
                    	 * (del_x * del_x + del_y * del_y + del_z * del_z
                       				- r_prime * r_prime));

            sim::time check;
            if (del_x * del_vx >= 0 && del_y * del_vy >= 0 &&
                del_z * del_vz >= 0){
                check = 0;
            }

            else if (rad_d > 0){
                check = 0;
            }

            else {
                check = (-(del_vx * del_x + del_vy * del_y + del_vz * del_z)
                        + sqrt(rad_d)) / (2 * (del_vx * del_vx + del_vz * del_vz
                        + del_vy * del_vy));
            }


            // Step 2 -- update list
            if (check != 0){
                test.rtime = check;
                test.part1 = i;
                test.part2 = j;
                list.push_back(test);
            }
            j++;
        }
        i++;
    }

    // Step 3 -- sort the list
		// Our lambda expression acts as a rule for std::sort
		// It will go through the list however the algorithm works
		// And re-sort it so that lowest rtime is @ begin
		// and highest @ end
		// So it automatically makes those changes to input list
		std::sort(list.begin(), list.end(),
		[](const Interaction &il, const Interaction &ir)
		{
			   return il.rtime < ir.rtime;
		}
		);
/*
    // std::sort()
    vector <Interaction> dummy;

    for (auto &ele : list){
    }

std::sort(std::begin(vec), std::end(vec), [](const Particle
                &lhs, const Particle &rhs){ return lhs.ts < rhs.ts; });
*/
    return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
void simulate(std::vector<int> interactions, std::vector<Particle> curr_data){


}
