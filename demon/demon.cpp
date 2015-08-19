/*------------demon.cpp-----------------------------------------------------//
*
*             demon.cpp --  We are evil. Join us. We will corrupt the world,
*                           one arrow of time at a time.
*
* Purpose: Segregate particles based on maxwell's demon.
*
*   Notes: This simulation will predominantly follow the logic of the link in
*          the README file
*
*          This file was written live by Leios on twitch.tv/simuleios
*          Please note that the efficiency of this code cannot be guaranteed.
*
*   ERROR: simulate function in infinite loop, make_list remains untested for
*              simulate function. New function mau be written in the future.
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include <cassert>

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct Particle{
    int pid, type;
    double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, time, mass, rad;
};

// holds interaction data
struct Interaction{
    double rtime;
    int part1, part2;
};

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel,
                               double max_vel2, double radius, double rad_2, 
                               double mass_1, double mass_2, double ratio);

// Makes the list for our simulation later, required starting data
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, int pnum, 
                                   const std::vector <int> &parts, int flag, 
                                   std::vector<Interaction> &list);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(std::vector<Interaction> &interactions, 
              std::vector<Particle> &curr_data, 
              double box_length, int pnum,
              double timeres, std::ofstream &output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(void){

    // opens file for writing
    std::ofstream output("out.dat", std::ofstream::out);

    int pnum = 100; // must be divisible by 1/ratio
    int type = 0;
    double box_length = 10, max_vel = .1, max_vel2 = .1, rad_1 = 0.1;
    double rad_2 = 0.01, timeres = 1, mass = 0.1, mass_2 = 0.01, ratio = 0.25;
    std::vector<int> parts(pnum);

    std::cout << "populate" << '\n';
    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel, 
                                               max_vel2, rad_1, rad_2, mass, 
                                               mass_2, ratio);

    std::cout << "the heck?" << '\n';

    for (const auto &p : curr_data){

        std::cout << p.pos_x << '\t' << p.pid << std::endl;
       
    }

    std::cout << "is the list right?" << '\n';
    std::vector<Interaction> list(pnum);
    std::cout << "before" << std::endl;
    list = make_list(curr_data, box_length, pnum, parts, type, list);
    std::cout << "after list made" << std::endl;

    //std::cout << std::endl << std::endl;

/*
    int count = 0;
    for (const auto &it : list){
        std::cout << count << '\t' << it.rtime << '\t' << it.part1  
                  << '\t' << it.part2 << std::endl;
        count++;
        
    }
*/

    simulate(list, curr_data, box_length, pnum, timeres, output);

    output.close();

    return 0;

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel,
                               double max_vel2, double radius, double rad_2,
                               double mass_1, double mass_2, double ratio){

    int pid_counter = 0;
    std::vector<Particle> curr_data(pnum);
    std::vector<Particle> left_data(pnum * ratio), right_data(pnum * (1-ratio));

    std::cout << pnum * 0.25 << '\t' << pnum * 0.75 << '\n';

    // static to create only 1 random_device
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);

    std::cout << seed << " Check this out!" << '\n';

    /* instead of doing % and * to get the correct distribution we directly
       specify which distribution we want */

    std::uniform_real_distribution<double> 
        box_length_distribution(radius, box_length - radius);

    std::uniform_real_distribution<double>
        box_left_distribution(rad_2 - box_length, rad_2);

    std::uniform_real_distribution<double> 
       max_vel_distribution(-max_vel, max_vel);

    std::uniform_real_distribution<double>
       max_velleft_distribution(-max_vel2, max_vel2);

    // For the left side of the box =)
    for (auto &p : left_data){ //read: for all particles p in left_data
        p.time = 0;
        p.type = 0;
        p.pid = pid_counter++;
        p.mass = mass_2;
        p.rad = rad_2;

        bool incorrectGeneration = true;
        while (incorrectGeneration){
            incorrectGeneration = false;
            for (auto &pos : { &p.pos_x }){
                *pos = box_left_distribution(gen);
            }

            for (auto &pos : { &p.pos_y, &p.pos_z }){
                *pos = box_length_distribution(gen);
            }

            for (int i = 0; i < p.pid; i++){
                auto& other_p = curr_data[i];
                // Distance < Radius * 2 Particle
                // Distance^2 < (Radius*2)^2
                if ( ((other_p.pos_x - p.pos_x)*(other_p.pos_x - p.pos_x) +
                    (other_p.pos_y - p.pos_y)*(other_p.pos_y - p.pos_y) +
                    (other_p.pos_z - p.pos_z)*(other_p.pos_z - p.pos_z))
                     < ((2*rad_2)*(2*rad_2)) ) {
                    incorrectGeneration = true;
                }
            }
        }
        
        for (auto &vel : { &p.vel_x, &p.vel_y, &p.vel_z }){
            *vel = max_velleft_distribution(gen);
        }
    }

    // For the Right data in the box
    for (auto &p : right_data){ //read: for all particles p in left_data
        p.time = 0;
        p.type = 1;
        p.pid = pid_counter++;
        p.mass = mass_1;
        p.rad = radius;

        bool incorrectGeneration = true;
        while (incorrectGeneration){
            incorrectGeneration = false;
            for (auto &pos : { &p.pos_x, &p.pos_y, &p.pos_z }){
                *pos = box_length_distribution(gen);
            }

            for (int i = 0; i < p.pid; i++){
                auto& other_p = curr_data[i];
                // Distance < Radius * 2 Particle
                // Distance^2 < (Radius*2)^2
                if ( ((other_p.pos_x - p.pos_x)*(other_p.pos_x - p.pos_x) +
                    (other_p.pos_y - p.pos_y)*(other_p.pos_y - p.pos_y) +
                    (other_p.pos_z - p.pos_z)*(other_p.pos_z - p.pos_z))
                     < ((2*radius)*(2*radius)) ) {
                    incorrectGeneration = true;
                }
            }
        }
        
        for (auto &vel : { &p.vel_x, &p.vel_y, &p.vel_z }){
            *vel = max_vel_distribution(gen);
        }
    }


    std::cout << "got to here" << '\n';
    // now concatenating the two!
    for (size_t ip = 0; ip < curr_data.size(); ip++){
        if (ip < pnum * ratio){
            curr_data[ip] = left_data[ip];
        }

        else{
            curr_data[ip] = right_data[ip - pnum * ratio];
        }
        std::cout << ip << '\n';
    }
    std::cout << "got past the loop of terror" << '\n';

    for (size_t i = 0; i < curr_data.size(); i++){
        std::cout << curr_data[i].pos_x << '\t' << i << std::endl;
    }

    std::cout << "is this our problem?" << '\n';

    return curr_data;
}

// Makes the list for our simulation later, required starting data
// Step 1: Check interactions between Particles, based on README link
// Step 2: Update list.
// Step 3: Sort list, lowest first
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, int pnum, 
                                   const std::vector<int> &parts, int flag, 
                                   std::vector<Interaction> &list){

    /* passing curr_data as const reference to avoid the copy and accidental
    overwriting */
    Interaction test;
    std::vector<Interaction> walltime(6);
    double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_tot, rad_d, del_vtot;
    double vdotr, taptime, epsilon = 0.001;
    int count = 0, ip;

    // Step 1 -- find interactions
    for (int i = 0; i < pnum; i++){

        //ip = parts[i];
        ip = i;

        if (ip >= 0){
            list[i].rtime = std::numeric_limits<double>::infinity();
            for (int k = 0; k < 6; k++ ){
                // setting arbitrarily high... 
                walltime[k].rtime = std::numeric_limits<double>::infinity();
                walltime[k].part1 = ip;
                walltime[k].part2 = - k - 1;
                count++;
            }
    
            // checking for interactions with the wall.
            // We need to take into account the velocity of the piston

            // case -1

            if (curr_data[ip].vel_x > 0){
                if (flag > 0){
                    if (curr_data[ip].pos_x < 0){

                        if (curr_data[ip].type == 0){
                            walltime[0].rtime = ( - curr_data[ip].pos_x
                                                  + curr_data[ip].rad)
                                                  / curr_data[ip].vel_x;
                        }

                        else{
                            walltime[0].rtime = 
                                std::numeric_limits<double>::infinity();
                        }

                    }
                    if (curr_data[ip].pos_x > 0){
                        if (curr_data[ip].type == 1){
                            walltime[0].rtime = (box_length-curr_data[ip].pos_x
                                                 + curr_data[ip].rad)
                                                 / curr_data[ip].vel_x;

                        }
                        else{
                            walltime[0].rtime = 
                                std::numeric_limits<double>::infinity();
                        }


                    }

                }
                else{
                    walltime[0].rtime = (box_length - curr_data[ip].pos_x
                                         -curr_data[ip].rad)
                                        / curr_data[ip].vel_x;
                }
            }

            // case -2
    
            if (curr_data[ip].vel_x < 0){
                if (flag > 0){
                    if (curr_data[ip].pos_x > 0){
                        walltime[1].rtime = ( - curr_data[ip].pos_x
                                             + curr_data[ip].rad)
                                             / curr_data[ip].vel_x;

                    }
                    if (curr_data[ip].pos_x < 0){
                        walltime[1].rtime = (-box_length - curr_data[ip].pos_x
                                             + curr_data[ip].rad)
                                             / curr_data[ip].vel_x;

                    }

                }
                else{
                    walltime[1].rtime = (-box_length - curr_data[ip].pos_x
                                         + curr_data[ip].rad)
                                         / curr_data[ip].vel_x;
                }
            }

            // case -3
    
            if (curr_data[ip].vel_y > 0){
                walltime[2].rtime = (box_length - curr_data[ip].pos_y
                                     - curr_data[ip].rad)
                                    / curr_data[ip].vel_y;
            }
    
            // case -4

            if (curr_data[ip].vel_y < 0){
                walltime[3].rtime = (-curr_data[ip].pos_y + curr_data[ip].rad)
                                    / curr_data[ip].vel_y;
            }
    
            // case -5

            if (curr_data[ip].vel_z > 0){
                walltime[4].rtime = (box_length - curr_data[ip].pos_z
                                     - curr_data[ip].rad)
                                    / curr_data[ip].vel_z;
            }
    
            // case -6

            if (curr_data[ip].vel_z < 0){
                walltime[5].rtime = (-curr_data[ip].pos_z + curr_data[ip].rad)
                                    / curr_data[ip].vel_z;
            }
    
            std::sort(std::begin(walltime), std::end(walltime),
                      [](const Interaction &dum1, const Interaction &dum2)
                         {return dum1.rtime < dum2.rtime;});
    
            for (int j = 0; j < pnum; j++){
                if (ip != j){
    
                    // simple definitions to make things easier later
                    del_x = curr_data[ip].pos_x - curr_data[j].pos_x;
                    del_y = curr_data[ip].pos_y - curr_data[j].pos_y;
                    del_z = curr_data[ip].pos_z - curr_data[j].pos_z;
    
                    del_vx = curr_data[ip].vel_x - curr_data[j].vel_x;
                    del_vy = curr_data[ip].vel_y - curr_data[j].vel_y;
                    del_vz = curr_data[ip].vel_z - curr_data[j].vel_z;
    
                    r_tot = curr_data[ip].rad + curr_data[j].rad;
    
                    // change in velocity * change in r
                    vdotr = del_vx*del_x + del_vy*del_y + del_vz*del_z;
                    del_vtot = del_vx*del_vx + del_vy*del_vy + del_vz*del_vz;
    
                    // This is under the radical in quad eq., thus "rad_d"
                    rad_d = (vdotr * vdotr) -  del_vtot
                            * (del_x*del_x + del_y*del_y + del_z*del_z 
                               - r_tot*r_tot);
    
                    if (vdotr >=0){
                        taptime = 0;
                    }
    
                    else if (rad_d < 0){
                        taptime = 0;
                    }
    
                    else {
                        taptime = (-(vdotr) + sqrt(rad_d)) / (del_vtot);
                    }
    
                    // Step 2 -- update list
                    if (taptime > 0 && taptime < walltime[0].rtime &&
                        taptime < list[ip].rtime){
    
                        //std::cout << "found one!" << std::endl;
                        test.rtime = taptime;
                        test.part1 = ip;
                        test.part2 = curr_data[j].pid;
                        list[ip] = test;
                    }
                }
    
            }

            if (walltime[0].rtime < list[i].rtime || list[i].rtime == 0){
                list[i] = walltime[0];
            }
/*    
            std::cout << '\n';

            std::cout << list[ip].rtime << '\t' << list[ip].part1 << '\t' 
                      << list[ip].part2 << '\n'; 
*/
    
        }

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


    std::cout << '\n' << '\n';

/*
    for (auto &p : list){

        std::cout << p.rtime << '\t' << p.part1 << '\t' 
                  << p.part2 << '\n';

    }
*/


    return list;
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
// Step 1: loop over timesteps, defined in list
// Step 2: model the interaction of particles that interact
// Step 3: update list
// Step 4: output positions to file
// UNCHECKED -- CERTAINLY BUG => infinite loop
void simulate(std::vector<Interaction> &interactions, 
              std::vector<Particle> &curr_data, 
              double box_length, int pnum,
              double timeres, std::ofstream &output){

    std::vector <int> teract(2);
    double del_x, del_y, del_z, J_x, J_y, J_z, J_tot, rtot, sigma, mass_tot;
    double del_vx, del_vy, del_vz, del_vtot, simtime = 0, tdiff;
    double timestep = 0, excess, first;
    int on = 1, count = 0, flag = 0;
    Particle temp = curr_data[on];

    // Note that these are all defined in the material linked in README
    // Step 1
    double half_time = 10 * interactions[80].rtime, final_time = 2 * half_time;

    while ( simtime < final_time ){
 

        std::cout << interactions[0].rtime << '\n';
        if (interactions[0].rtime < 0){

            std::cout << interactions[0].rtime << '\n';
            std::cout << interactions[0].part2 << '\t'
                      << interactions[0].part1 << '\t' 
                      << curr_data[interactions[0].part1].pos_x << '\t' 
                      << curr_data[interactions[0].part1].pos_y << '\t' 
                      << curr_data[interactions[0].part1].pos_z << '\t' 
                      << curr_data[interactions[0].part1].vel_x << '\t' 
                      << curr_data[interactions[0].part1].vel_y << '\t' 
                      << curr_data[interactions[0].part1].vel_z << '\t' 
                      << curr_data[interactions[0].part1].rad << '\t'
                      << half_time / 5 << '\t' << flag << '\n';

            assert(interactions[0].rtime > 0);
        }

        if (interactions[0].rtime < 5e-10){

            std::cout << interactions[0].rtime << '\n';
            std::cout << interactions[0].part2 << '\t'
                      << interactions[0].part1 << '\t' 
                      << curr_data[interactions[0].part1].pos_x << '\t' 
                      << curr_data[interactions[0].part1].pos_y << '\t' 
                      << curr_data[interactions[0].part1].pos_z << '\t' 
                      << curr_data[interactions[0].part1].vel_x << '\t' 
                      << curr_data[interactions[0].part1].vel_y << '\t' 
                      << curr_data[interactions[0].part1].vel_z << '\t' 
                      << curr_data[interactions[0].part1].rad << '\t'
                      << half_time / 5 << '\t' << flag << '\n';

            assert(interactions[0].rtime > 5e-10);
        }

        std::cout << "simtime is: " << simtime << '\n';
        std::cout << "the particles are: " << interactions[0].part1 << '\t'
                  << interactions[0].part2 << '\t'
                  << curr_data[interactions[0].part1].pos_x << '\t'
                  << curr_data[interactions[0].part1].pos_y << '\t' 
                  << curr_data[interactions[0].part1].pos_z << '\t' 
                  << curr_data[interactions[0].part1].vel_x << '\t' 
                  << curr_data[interactions[0].part1].vel_y << '\t' 
                  << curr_data[interactions[0].part1].vel_z << '\t' 
                  << curr_data[interactions[0].part1].rad << '\n';


        // output data from previous time interaction step
        // First, update the velocities of out temp vector and define a count
        // so we can specially calculate the first step of our loop.
        temp.vel_x = curr_data[on].vel_x;
        temp.vel_y = curr_data[on].vel_y;
        temp.vel_z = curr_data[on].vel_z;
        count = 0;

        // output the interaction with the wall.
        temp.pos_x = curr_data[on].pos_x;
        temp.pos_y = curr_data[on].pos_y;
        temp.pos_z = curr_data[on].pos_z;

        output << temp.pid << '\t' << simtime << '\t' 
               << temp.pos_x << '\t'
               << temp.pos_y << '\t'
               << temp.pos_z << '\t'
               << temp.vel_x << '\t'
               << temp.vel_y << '\t' << temp.vel_z
               << '\n';

        // simtime has yet to be updated, so we are running until our timestep
        // values are *just* under our next simtime value, but we are keeping
        // the old value for the first loop step


        while (timestep < simtime + interactions[0].rtime){

            if (count == 0){

                excess = timestep - simtime;

                temp.pos_x += temp.vel_x * excess;
                temp.pos_y += temp.vel_y * excess;
                temp.pos_z += temp.vel_z * excess;

                output << temp.pid << '\t' << timestep << '\t' 
                       << temp.pos_x << '\t'
                       << temp.pos_y << '\t'
                       << temp.pos_z << '\t'
                       << temp.vel_x << '\t'
                       << temp.vel_y << '\t' << temp.vel_z
                       << '\n';

            }

            else{
                temp.pos_x += temp.vel_x*(timeres);
                temp.pos_y += temp.vel_y*(timeres);
                temp.pos_z += temp.vel_z*(timeres);

                output << temp.pid << '\t' << timestep << '\t' 
                       << temp.pos_x << '\t'
                       << temp.pos_y << '\t'
                       << temp.pos_z << '\t'
                       << temp.vel_x << '\t'
                       << temp.vel_y << '\t' << temp.vel_z
                       << '\n';
 
            }
            count++;
            timestep += timeres;

        }

        // Radioactive

        // Now to output the position of the wall

        // Now we are updating simtime and moving on with interactions and such

        tdiff = interactions[0].rtime;
        simtime += interactions[0].rtime;

        // Now let's update our interactions list
/*
        first = interactions[0].rtime;
        for ( int q = 0; q < pnum; q++){
            interactions[q].rtime -= first;
            //std::cout << interactions[q].rtime << '\n';
        }
*/

        teract[0] = interactions[0].part1;
        teract[1] = interactions[0].part2;

        // updating positions in curr_data
        for ( int i = 0; i < pnum; i++){
            curr_data[i].pos_x += curr_data[i].vel_x * tdiff;
            curr_data[i].pos_y += curr_data[i].vel_y * tdiff;
            curr_data[i].pos_z += curr_data[i].vel_z * tdiff;

/*
            output << curr_data[i].pid << '\t' << simtime << '\t' 
                   << curr_data[i].pos_x << '\t' << curr_data[i].pos_y << '\t'
                   << curr_data[i].pos_y << '\t' << curr_data[i].vel_x << '\t'
                   << curr_data[i].vel_y << '\t' << curr_data[i].vel_z << '\t'
                   << '\n';
*/
        }

        //std::cout << simtime << '\n'<< '\n';

        if (interactions[0].part2 > 0){
            del_x = (curr_data[interactions[0].part1].pos_x) 
                     - (curr_data[interactions[0].part2].pos_x); 

	    del_y = (curr_data[interactions[0].part1].pos_y) 
                     - (curr_data[interactions[0].part2].pos_y); 

            del_z = (curr_data[interactions[0].part1].pos_z) 
                     - (curr_data[interactions[0].part2].pos_z); 
    
            del_vx = (curr_data[interactions[0].part1].vel_x)
                     - (curr_data[interactions[0].part2].vel_x);
    
            del_vy = (curr_data[interactions[0].part1].vel_y)
                     - (curr_data[interactions[0].part2].vel_y);
    
            del_vz = (curr_data[interactions[0].part1].vel_z)
                     - (curr_data[interactions[0].part2].vel_z);
    
    
            rtot = (del_x*del_x + del_y*del_y + del_z*del_z);
            del_vtot = (del_vx*del_vx + del_vy*del_vy + del_vz*del_vz);
            sigma = curr_data[interactions[0].part1].rad + 
                    curr_data[interactions[0].part2].rad;
            mass_tot = curr_data[interactions[0].part1].mass +
                       curr_data[interactions[0].part2].mass;
    
            // Step 2
            J_tot = ( curr_data[interactions[0].part1].mass *
                         curr_data[interactions[0].part2].mass *
                         del_vtot * rtot) / (sigma * mass_tot);
    
            J_x = J_tot * del_x / sigma;
            J_y = J_tot * del_y / sigma;
            J_z = J_tot * del_z / sigma;
    
            curr_data[interactions[0].part1].vel_x += J_x
            / curr_data[interactions[0].part1].mass;
            curr_data[interactions[0].part1].vel_y += J_y
            / curr_data[interactions[0].part1].mass;
            curr_data[interactions[0].part1].vel_z += J_z
            / curr_data[interactions[0].part1].mass;
    
            curr_data[interactions[0].part2].vel_x -= J_x
            / curr_data[interactions[0].part2].mass;
            curr_data[interactions[0].part2].vel_y -= J_y
            / curr_data[interactions[0].part2].mass;
            curr_data[interactions[0].part2].vel_z -= J_z
            / curr_data[interactions[0].part2].mass;
            //std::cout << "check_interaction" << '\n';

        if (abs(curr_data[interactions[0].part1].vel_x) > 1000 ||
            abs(curr_data[interactions[0].part1].vel_y) > 1000 ||
            abs(curr_data[interactions[0].part1].vel_z) > 1000 ||
            abs(curr_data[interactions[0].part2].vel_x) > 1000 ||
            abs(curr_data[interactions[0].part2].vel_y) > 1000 ||
            abs(curr_data[interactions[0].part2].vel_z) > 1000 ){

            std::cout << interactions[0].rtime << '\n';
            std::cout << interactions[0].part2 << '\t'
                      << interactions[0].part1 << '\t' 
                      << curr_data[interactions[0].part1].pos_x << '\t' 
                      << curr_data[interactions[0].part1].pos_y << '\t' 
                      << curr_data[interactions[0].part1].pos_z << '\t' 
                      << curr_data[interactions[0].part1].vel_x << '\t' 
                      << curr_data[interactions[0].part1].vel_y << '\t' 
                      << curr_data[interactions[0].part1].vel_z << '\t' 
                      << curr_data[interactions[0].part1].rad << '\t'
                      << J_tot << '\t' << J_x << '\t' << J_y << '\t' << J_z 
                      << '\n';

            assert(abs(curr_data[interactions[0].part1].vel_x) < 1000);
            assert(abs(curr_data[interactions[0].part1].vel_y) < 1000);
            assert(abs(curr_data[interactions[0].part1].vel_z) < 1000);
            assert(abs(curr_data[interactions[0].part2].vel_x) < 1000);
            assert(abs(curr_data[interactions[0].part2].vel_y) < 1000);
            assert(abs(curr_data[interactions[0].part2].vel_z) < 1000);

        }

        }

        if (interactions[0].part2 < 0){
            //std::cout << "wall check" << std::endl;
            switch(interactions[0].part2){
                case -1:
                    curr_data[interactions[0].part1].vel_x *= -1.0;
                    //std::cout << curr_data[interactions[0].part1].pos_x <<'\t'
                    //          << -1 << '\n';

                    break;
                
                case -2:
                    std::cout << curr_data[interactions[0].part1].vel_x << '\n';
                    curr_data[interactions[0].part1].vel_x *= -1.0; 
                    std::cout << curr_data[interactions[0].part1].vel_x <<'\t'
                              << -2 << '\n';

                    break;
                
                case -3:
                    curr_data[interactions[0].part1].vel_y *= -1.0;
                    //std::cout << curr_data[interactions[0].part1].pos_y <<'\t'
                    //          << -3 << '\n';
                    break;
                
                case -4:
                    curr_data[interactions[0].part1].vel_y *= -1.0;
                    //std::cout << curr_data[interactions[0].part1].pos_y <<'\t'
                    //          << -4 << '\n';
                    break;
                
                case -5:
                    curr_data[interactions[0].part1].vel_z *= -1.0;
                    //std::cout << curr_data[interactions[0].part1].pos_z <<'\t'
                    //          << -5 << '\n';
                    break;
                
                case -6:
                    curr_data[interactions[0].part1].vel_z *= -1.0;
                    //std::cout << curr_data[interactions[0].part1].pos_z <<'\t'
                    //          << -6 << '\n';
                    break;

            }

        
        }

        if (simtime > half_time ){ // &&
            //simtime < half_time + interactions[0].rtime){
            flag = 1;

            for (int q = 0; q < pnum; q++){
                del_x = (abs(curr_data[q].pos_x) - curr_data[q].rad);

                if (del_x <= 0){
                    if (curr_data[q].pos_x >= 0){
                        curr_data[q].pos_x += curr_data[q].rad;
                    }
                    if (curr_data[q].pos_x < 0){
                        curr_data[q].pos_x -= curr_data[q].rad;
                    }

                }
            
            }
        }


        // Step 3 -- update list; TODO
        // UNCHECKED
        interactions = make_list(curr_data, box_length, pnum,
                                 teract, flag, interactions);
        
/*
        for (const auto &it : interactions){
            std::cout << it.rtime << std::endl;
        }
*/

    }

}
