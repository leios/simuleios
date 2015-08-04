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
#include <algorithm>

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

    int pnum = 1000;
    double box_length = 10, max_vel = 0.01;

    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel);

    for (auto &p : curr_data){

        std::cout << p.pos_x << '\n';

    }

    std::vector<Interaction> list = make_list(curr_data, box_length, 0.1);

    std::cout << '\n' << '\n';

    for (auto &it : list){
        std::cout << it.rtime << '\n';
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

    // Step 1 -- find interactions
    /* The "for (auto &e : c)" syntax is not very useful when we need the
    index and don't iterate over every element */
    /* After we compute the pair i-j we don't need to compare the pair
    j-i anymore, also no i-i comparison, so make j > i */
    for (size_t i = 0; i < curr_data.size(); ++i){
        for (size_t j = i + 1; j < curr_data.size(); ++j){
            //think of this as giving "curr_data[i]" the new name "ip"
            const auto &ip = curr_data[i];
            const auto &jp = curr_data[j];

            const double delta_x = ip.pos_x - jp.pos_x;
            const double delta_y = ip.pos_y - jp.pos_y;
            const double delta_z = ip.pos_z - jp.pos_y;

            const double delta_vx = ip.vel_x - jp.vel_y;
            const double delta_vy = ip.vel_y - jp.vel_y;
            const double delta_vz = ip.vel_z - jp.vel_z;

            if (delta_x * delta_vx >= 0 && delta_y * delta_vy >= 0 &&
                delta_z * delta_vz >= 0){
                continue;
            }

            const double r_tot = 2 * radius;
            //change "r_tot" and "rad_d" to a more descriptive name?
            const double delta = delta_vx*delta_x + delta_vy*delta_y + delta_vz*delta_z;
            const double rad_d = (delta * delta
                - 4 * (delta_vx*delta_vx + delta_vy*delta_vy + delta_vz*delta_vz)
                * (delta_x*delta_x + delta_y*delta_y + delta_z*delta_z
                - r_tot*r_tot));

            // NaN error here! Sorry about that ^^
            if (rad_d < 0){
                continue;
            }

            // Step 2 -- update list
            std::cout << "found one!" << '\n';
            const auto check = (-(delta_vx*delta_x + delta_vy*delta_y + delta_vz*delta_z)
                + sqrt(rad_d)) / (2 *
                (delta_vx*delta_vx + delta_vz*delta_vz + delta_vy*delta_vy));
            Interaction test;
            test.rtime = check;
            test.part1 = i;
            test.part2 = j;
            list.push_back(test);
        }
    }

    // Step 3 -- sort the list by rtime
    /* given 2 objects, std::sort needs to know if the first one is smaller */
    std::sort(std::begin(list), std::end(list),
        [](const Interaction &lhs, const Interaction &rhs){
            return lhs.rtime < rhs.rtime;
        }
    );
    return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
void simulate(std::vector<int> interactions, std::vector<Particle> curr_data){


}

