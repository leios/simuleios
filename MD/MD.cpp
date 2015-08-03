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
//#include <algorithm>

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

    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel);

    for (auto &p : curr_data){

        std::cout << p.pos_x << std::endl;
       
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
    double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_prime, rad_d;

    // Step 1 -- find interactions
    for (auto &ip : curr_data){
        for (auto &jp : curr_data){
            del_x = ip.pos_x - jp.pos_x;
            del_y = ip.pos_y - jp.pos_y;
            del_z = ip.pos_z - jp.pos_y;

            del_vx = ip.vel_x - jp.vel_y;
            del_vy = ip.vel_y - jp.vel_y;
            del_vz = ip.vel_z - jp.vel_z;

            r_prime = 2 * radius;

            rad_d = (pow(del_vx * del_x + del_vy * del_y 
                                 + del_vz * del_z, 2)
                    - 4 * (del_vx * del_vx + del_vy * del_vy + del_vz * del_vz)
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

