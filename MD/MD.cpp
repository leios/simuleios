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
*          This file was written live by Leios on twitch.tv/simuleios
*          Please note that the efficiency of this code cannot be guaranteed.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct Particle{
    int pid;
    double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, time;
};

// holds interaction data
struct Interaction{
    double rtime;
    int part1, part2;
};

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel);

// Makes the list for our simulation later, required starting data
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, double radius);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(std::vector<Interaction> &interactions, 
              std::vector<Particle> &curr_data, 
              double radius, double mass, double box_length);

// Update list during simulate
std::vector<Interaction> update_list(const std::vector<Particle> &curr_data,
                                     double box_length, double radius,
                                     std::vector<Interaction> list);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(void){

    int pnum = 10000;
    double box_length = 10, max_vel = 0.01;

    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel);

    for (const auto &p : curr_data){

        std::cout << p.pos_x << std::endl;
       
    }

    std::vector<Interaction> list = make_list(curr_data, box_length,0.1);

    std::cout << std::endl << std::endl;

    for (const auto &it : list){
        std::cout << it.rtime << std::endl;
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

    int pid_counter = 0;
    for (auto &p : curr_data){ //read: for all particles p in curr_data
        p.time = 0;
        p.pid = pid_counter++;

        for (auto &pos : { &p.pos_x, &p.pos_y, &p.pos_z }){
            *pos = box_length_distribution(gen);
        }

        for (auto &vel : { &p.vel_x, &p.vel_y, &p.vel_z }){
            *vel = max_vel_distribution(gen);
        }
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
    int i = 0, j;
    double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_tot, rad_d, del_vtot;

    // Step 1 -- find interactions
    for (const auto &ip : curr_data){
        j = 0;
        for (const auto &jp : curr_data){
            if (i != j){

                // simple definitions to make things easier later
                del_x = ip.pos_x - jp.pos_x;
                del_y = ip.pos_y - jp.pos_y;
                del_z = ip.pos_z - jp.pos_y;

                del_vx = ip.vel_x - jp.vel_y;
                del_vy = ip.vel_y - jp.vel_y;
                del_vz = ip.vel_z - jp.vel_z;
                
				// sigma ?
                r_tot = 2 * radius;

				// This is actually the sqrt(del_vtot)
				// del_v * del_r ?
				del_vtot = del_vx*del_x + del_vy*del_y + del_vz*del_z;
				// del_r * del_ r = del_x^2 + del_y^2 + del_z^2 ?
				// del_v * delv_v = del_vx^2 + del_vy^2 + del_vz^2 ?

				// This is under the radical in quad eq., thus "rad_d"
				// d ?
				rad_d = (del_vtot * del_vtot
					- (del_vx*del_vx + del_vy*del_vy + del_vz*del_vz)
					* (del_x*del_x + del_y*del_y + del_z*del_z
					- r_tot*r_tot));

				double taptime;
				if (del_vtot >= 0){
					taptime = 0;
				}

				// NaN error here! Sorry about that ^^, lt was gt (oops!)
				else if (rad_d < 0){
					taptime = 0;
				}

				else {
					taptime = -(del_vtot + sqrt(rad_d)) /
						(del_vx*del_vx + del_vy*del_vy + del_vz*del_vz);
				}


                // Step 2 -- update list
                if (taptime != 0){
                    std::cout << "found one!" << std::endl;
                    test.rtime = taptime;
                    test.part1 = i;
                    test.part2 = j;
                    list.push_back(test);
                }
            }
            j++;
        }
        i++;
    }

    // Step 3 -- sort the list

    // The std::sort command wants to know 3 things:
    //     1. Where to start sorting
    //     2. Where to stop sorting
    //     3. Which element value is greater
    // 
    // To check #3, we create a lambda (on-the-fly) function that reads in the
    //     two previous variables as dum1 and dum2 and checks which element is
    //     lesser. That's it.

    std::sort(std::begin(list), std::end(list),
              [](const Interaction &dum1, const Interaction &dum2)
                 {return dum1.rtime < dum2.rtime;});

    return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
// Step 1: loop over timesteps, defined in list
// Step 2: model the interaction of particles that interact
// Step 3: update list
// Step 4: output positions to file
// UNCHECKED
void simulate(std::vector<Interaction> &interactions, 
              std::vector<Particle> &curr_data, 
              double radius, double mass, double box_length){

    double del_x, del_y, del_z, J_x, J_y, J_z, J_tot, rtot;
    double del_vx, del_vy, del_vz, del_vtot;

    // Note that these are all defined in the material linked in README
    // Step 1
    for (double simtime = 0; simtime < interactions.back().rtime; 
         simtime += interactions[0].rtime){

        del_x = (curr_data[interactions[0].part1].pos_x 
                 + curr_data[interactions[0].part1].vel_x * simtime)
                 - (curr_data[interactions[0].part2].pos_x 
                 + curr_data[interactions[0].part2].vel_x * simtime);

        del_y = (curr_data[interactions[0].part1].pos_y 
                 + curr_data[interactions[0].part1].vel_y * simtime)
                 - (curr_data[interactions[0].part2].pos_y 
                 + curr_data[interactions[0].part2].vel_y * simtime);

        del_z = (curr_data[interactions[0].part1].pos_z 
                 + curr_data[interactions[0].part1].vel_z * simtime)
                 - (curr_data[interactions[0].part2].pos_z 
                 + curr_data[interactions[0].part2].vel_z * simtime);

        del_vx = (curr_data[interactions[0].part1].vel_x)
                 - (curr_data[interactions[0].part2].vel_x);

        del_vy = (curr_data[interactions[0].part1].vel_y)
                 - (curr_data[interactions[0].part2].vel_y);

        del_vz = (curr_data[interactions[0].part1].vel_z)
                 - (curr_data[interactions[0].part2].vel_z);


        rtot = sqrt(del_x * del_x + del_y * del_y + del_z * del_z);
        del_vtot = sqrt(del_vx * del_vx + del_vy * del_vy + del_vz * del_vz);

        // Step 2
        J_tot = (2 * mass * mass * del_vtot * rtot) / (2 * radius * (2 * mass));

        J_x = J_tot * del_x / (2 * radius);
        J_y = J_tot * del_y / (2 * radius);
        J_z = J_tot * del_z / (2 * radius);

        curr_data[interactions[0].part1].vel_x += J_x / mass;
        curr_data[interactions[0].part1].vel_y += J_y / mass;
        curr_data[interactions[0].part1].vel_z += J_z / mass;

        curr_data[interactions[0].part2].vel_x -= J_x / mass;
        curr_data[interactions[0].part2].vel_y -= J_y / mass;
        curr_data[interactions[0].part2].vel_z -= J_z / mass;

        // Step 3 -- update list; TODO
        // UNCHECKED
        interactions = update_list(curr_data, box_length, radius, interactions);

    }


}

// Update list during simulate
// NOT FINISHED
std::vector<Interaction> update_list(const std::vector<Particle> &curr_data,
                                     double box_length, double radius,
                                     std::vector<Interaction> list){

    Interaction test;
    int i = 0;
    double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_tot, rad_d, del_vtot;

    // Copied from above in make_list
    for (const auto &ip : curr_data){
        for (int j = 0; j < 2; j++){
            if (i != 0){
                auto &jp = curr_data[list[j].part1];

                // simple definitions to make things easier later
                del_x = ip.pos_x - jp.pos_x;
                del_y = ip.pos_y - jp.pos_y;
                del_z = ip.pos_z - jp.pos_y;

                del_vx = ip.vel_x - jp.vel_y;
                del_vy = ip.vel_y - jp.vel_y;
                del_vz = ip.vel_z - jp.vel_z;

                r_tot = 2 * radius;

                // This is actually the sqrt(del_vtot)
                del_vtot = del_vx*del_x + del_vy*del_y + del_vz*del_z;

                // This is under the radical in quad eq., thus "rad_d"
                rad_d = (del_vtot * del_vtot
                        - 4 * (del_vx*del_vx + del_vy*del_vy + del_vz*del_vz)
                        * (del_x*del_x + del_y*del_y + del_z*del_z 
                           - r_tot*r_tot));

                double taptime;
                if (del_x * del_vx >= 0 && del_y * del_vy >= 0 &&
                    del_z * del_vz >= 0){
                    taptime = 0;
                }

                // NaN error here! Sorry about that ^^, lt was gt (oops!)
                else if (rad_d < 0){
                    taptime = 0;
                }

                else {
                    taptime = (-(del_vx*del_x + del_vy*del_y + del_vz*del_z)
                            + sqrt(rad_d)) / (2 * 
                            (del_vx*del_vx + del_vz*del_vz + del_vy*del_vy));
                }


                // Step 2 -- update list
                if (taptime != 0){
                    std::cout << "found one!" << std::endl;
                    test.rtime = taptime;
                    test.part1 = i;
                    test.part2 = j;
                    list.push_back(test);
                }
            }
        }
        i++;
    }

    std::sort(std::begin(list), std::end(list),
              [](const Interaction &dum1, const Interaction &dum2)
                 {return dum1.rtime < dum2.rtime;});

    return list;
}
