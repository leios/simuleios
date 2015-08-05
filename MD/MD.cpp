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
<<<<<<< HEAD
#include "Vec.h"

namespace sim{
    /* putting our stuff in its own namespace so we can call "time" "time"
    without colliding with names from the standard library */
=======

namespace sim{
    /* putting our stuff in its own namespace so we can call "time" "time"
       without colliding with names from the standard library */
>>>>>>> bcd5da5553db6a4ef9f94126e54cef9050810718
    using time = double;
}

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct Particle{
    int PID;
    sim::time ts;
<<<<<<< HEAD
    //double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;
    sim::Vec pos{ 0, 0, 0 }, vel{ 0, 0, 0 };
    //initializing pos and vel to 0 is not ideal
=======
    double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;
>>>>>>> bcd5da5553db6a4ef9f94126e54cef9050810718
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
void simulate(std::vector<Interaction> interactions,
    std::vector<Particle> curr_data,
    double radius, double mass, double box_length);

// Update list during simulate
std::vector<Interaction> update_list(const std::vector<Particle> &curr_data,
    double box_length, double radius,
    std::vector<Interaction> list);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(void){

    int pnum = 1000;
    double box_length = 10, max_vel = 0.01;

    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel);

    for (auto &p : curr_data){

<<<<<<< HEAD
        std::cout << p.pos.x << std::endl;
=======
        std::cout << p.pos_x << '\n';
>>>>>>> bcd5da5553db6a4ef9f94126e54cef9050810718

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

<<<<<<< HEAD
        for (auto &pos : { &p.pos.x, &p.pos.y, &p.pos.z }){
            *pos = box_length_distribution(gen);
        }

        for (auto &vel : { &p.vel.x, &p.vel.y, &p.vel.z }){
            *vel = max_vel_distribution(gen);
        }
=======
        for (auto &pos : { &p.pos_x, &p.pos_y, &p.pos_z })
            *pos = box_length_distribution(gen);
        for (auto &vel : { &p.vel_x, &p.vel_y, &p.vel_z })
            *vel = max_vel_distribution(gen);
>>>>>>> bcd5da5553db6a4ef9f94126e54cef9050810718
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
<<<<<<< HEAD
    Interaction test;
    int i = 0, j = 0;
    //double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_tot, rad_d, del_vtot;

    // Step 1 -- find interactions
    for (auto &ip : curr_data){
        for (auto &jp : curr_data){
            if (i != j){

                // simple definitions to make things easier later
                /*
                del_x = ip.pos_x - jp.pos_x;
                del_y = ip.pos_y - jp.pos_y;
                del_z = ip.pos_z - jp.pos_y;

                del_vx = ip.vel_x - jp.vel_y;
                del_vy = ip.vel_y - jp.vel_y;
                del_vz = ip.vel_z - jp.vel_z;
                */
                using sim::Vec;
                Vec del_p = ip.pos - jp.pos;
                Vec del_v = ip.vel - jp.vel;

                double r_tot = 2 * radius;

                // This is actually the sqrt(del_vtot)
                //del_vtot = del_vx*del_x + del_vy*del_y + del_vz*del_z;
                double del_vtot = (del_v * del_p).sum();

                // This is under the radical in quad eq., thus "rad_d"
                /*
                rad_d = (del_vtot * del_vtot
                    - 4 * (del_vx*del_vx + del_vy*del_vy + del_vz*del_vz)
                    * (del_x*del_x + del_y*del_y + del_z*del_z
                    - r_tot*r_tot));
                */
                double rad_d = (del_vtot * del_vtot
                    - 4 * (del_v * del_v).sum()
                    * ((del_p * del_p).sum()
                    - r_tot*r_tot));

                sim::time check;
                /*
                if (del_x * del_vx >= 0 && del_y * del_vy >= 0 &&
                    del_z * del_vz >= 0){
                    check = 0;
                }
                */
                const auto del_pv = del_p * del_v;
                if (del_pv.x >= 0 && del_pv.y >= 0 &&
                    del_pv.z >= 0){
                    check = 0;
                }


                // NaN error here! Sorry about that ^^, lt was gt (oops!)
                else if (rad_d < 0){
                    check = 0;
                }

                else {
                    /*
                    check = (-(del_vx*del_x + del_vy*del_y + del_vz*del_z)
                        + sqrt(rad_d)) / (2 *
                        (del_vx*del_vx + del_vz*del_vz + del_vy*del_vy));
                    */
                    check = (-(del_v * del_p).sum()
                        + sqrt(rad_d)) / (2 *
                        (del_v * del_v).sum());
                }


                // Step 2 -- update list
                if (check != 0){
                    std::cout << "found one!" << std::endl;
                    test.rtime = check;
                    test.part1 = i;
                    test.part2 = j;
                    list.push_back(test);
                }
=======

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
>>>>>>> bcd5da5553db6a4ef9f94126e54cef9050810718
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

<<<<<<< HEAD
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
    {return dum1.rtime < dum2.rtime; });

=======
    // Step 3 -- sort the list by rtime
    /* given 2 objects, std::sort needs to know if the first one is smaller */
    std::sort(std::begin(list), std::end(list),
        [](const Interaction &lhs, const Interaction &rhs){
            return lhs.rtime < rhs.rtime;
        }
    );
>>>>>>> bcd5da5553db6a4ef9f94126e54cef9050810718
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
void simulate(std::vector<Interaction>& interactions,
    std::vector<Particle>& curr_data,
    double radius, double mass, double box_length){

    //double del_x, del_y, del_z, J_x, J_y, J_z, J_tot, rtot;
    //double del_vx, del_vy, del_vz, del_vtot;

    // Note that these are all defined in the material linked in README
    // Step 1
    for (double simtime = 0; simtime < interactions.back().rtime;
        simtime += interactions[0].rtime){
        /*
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
        */
        using sim::Vec;
        Vec del_p = curr_data[interactions[0].part1].pos
            + curr_data[interactions[0].part1].vel * simtime
            - curr_data[interactions[0].part2].pos
            + curr_data[interactions[0].part2].vel * simtime;
        Vec del_v = curr_data[interactions[0].part1].vel
            - curr_data[interactions[0].part2].vel;

        //rtot = sqrt(del_x * del_x + del_y * del_y + del_z * del_z);
        double rtot = del_p.length();
        //del_vtot = sqrt(del_vx * del_vx + del_vy * del_vy + del_vz * del_vz);
        double del_vtot = del_v.length();

        // Step 2
        double J_tot = (2 * mass * mass * del_vtot * rtot) / (2 * radius * (2 * mass));

        /*
        J_x = J_tot * del_x / (2 * radius);
        J_y = J_tot * del_y / (2 * radius);
        J_z = J_tot * del_z / (2 * radius);
        */
        Vec J = J_tot * del_p / (2 * radius);

        /*
        curr_data[interactions[0].part1].vel_x += J_x / mass;
        curr_data[interactions[0].part1].vel_y += J_y / mass;
        curr_data[interactions[0].part1].vel_z += J_z / mass;

        curr_data[interactions[0].part2].vel_x -= J_x / mass;
        curr_data[interactions[0].part2].vel_y -= J_y / mass;
        curr_data[interactions[0].part2].vel_z -= J_z / mass;
        */

        curr_data[interactions[0].part1].vel += J / mass;
        curr_data[interactions[0].part2].vel += J / mass;

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
    //double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_tot, rad_d, del_vtot;

    // Copied from above in make_list
    for (auto &ip : curr_data){
        for (int j = 0; j < 2; j++){
            if (i != 0){
                auto &jp = curr_data[list[j].part1];

                // simple definitions to make things easier later
                /*
                del_x = ip.pos_x - jp.pos_x;
                del_y = ip.pos_y - jp.pos_y;
                del_z = ip.pos_z - jp.pos_y;

                del_vx = ip.vel_x - jp.vel_y;
                del_vy = ip.vel_y - jp.vel_y;
                del_vz = ip.vel_z - jp.vel_z;
                */
                using sim::Vec;
                Vec del_p = ip.pos - jp.pos;
                Vec del_v = ip.vel - jp.vel;

                double r_tot = 2 * radius;

                // This is actually the sqrt(del_vtot)
                //del_vtot = del_vx*del_x + del_vy*del_y + del_vz*del_z;
                double del_vtot = (del_v * del_p).sum();

                // This is under the radical in quad eq., thus "rad_d"
                /*
                rad_d = (del_vtot * del_vtot
                    - 4 * (del_vx*del_vx + del_vy*del_vy + del_vz*del_vz)
                    * (del_x*del_x + del_y*del_y + del_z*del_z
                    - r_tot*r_tot));
                */
                double rad_d = (del_vtot * del_vtot
                    - 4 * (del_v * del_v).sum()
                    * ((del_p * del_p).sum()
                    - r_tot*r_tot));

                sim::time check;
                /*
                if (del_x * del_vx >= 0 && del_y * del_vy >= 0 &&
                    del_z * del_vz >= 0){
                    check = 0;
                }
                */
                if (del_p.x * del_v.x >= 0 && del_p.y * del_v.y >= 0 &&
                    del_p.z * del_v.z >= 0){
                    check = 0;
                }

                // NaN error here! Sorry about that ^^, lt was gt (oops!)
                else if (rad_d < 0){
                    check = 0;
                }

                else {
                    /*
                    check = (-(del_vx*del_x + del_vy*del_y + del_vz*del_z)
                        + sqrt(rad_d)) / (2 *
                        (del_vx*del_vx + del_vz*del_vz + del_vy*del_vy));
                    */
                    check = (-(del_v * del_p).sum()
                        + sqrt(rad_d)) / (2 *
                        (del_v * del_v).sum());
                }


                // Step 2 -- update list
                if (check != 0){
                    std::cout << "found one!" << std::endl;
                    test.rtime = check;
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
    {return dum1.rtime < dum2.rtime; });



    return list;
}
