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
*   ERROR: simulate function in infinite loop, make_list remains untested for
*              simulate function. New function mau be written in the future.
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
std::vector<Particle> populate(int pnum, double box_length, double max_vel,
                               double radius);

// Makes the list for our simulation later, required starting data
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, double radius, int pnum);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(std::vector<Interaction> &interactions, 
              std::vector<Particle> &curr_data, 
              double radius, double mass, double box_length, int pnum);

// Update list during simulate
std::vector<Interaction> update_list(const std::vector<Particle> &curr_data,
                                     double box_length, double radius,
                                     std::vector<Interaction> list);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(void){

    // opens file for writing
    std::ofstream output("out.dat", std::ofstream::out);

    int pnum = 100;
    double box_length = 10, max_vel = 0.01, radius = 0.0001, mass = 0.1;

    std::vector<Particle> curr_data = populate(pnum, box_length, max_vel, 
                                               radius);

    for (const auto &p : curr_data){

        std::cout << p.pos_x << '\t' << p.pid << std::endl;
       
    }

    std::vector<Interaction> list = make_list(curr_data, box_length,0.1, pnum);

    std::cout << std::endl << std::endl;

    int count = 0;
    for (const auto &it : list){
        std::cout << count << '\t' << it.rtime << '\t' << it.part1  
                  << '\t' << it.part2 << std::endl;
        count++;
        
    }

    //simulate(list, curr_data, radius, mass, box_length, pnum);

    return 0;
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Makes starting data
std::vector<Particle> populate(int pnum, double box_length, double max_vel,
                               double radius){
    std::vector<Particle> curr_data(pnum);

    // static to create only 1 random_device
    static std::random_device rd;
    static std::mt19937 gen(rd());

    /* instead of doing % and * to get the correct distribution we directly
       specify which distribution we want */

    std::uniform_real_distribution<double> 
        box_length_distribution(radius, box_length - radius);
    std::uniform_real_distribution<double> 
       max_vel_distribution(0, max_vel);

    int pid_counter = 0;
    for (auto &p : curr_data){ //read: for all particles p in curr_data
        p.time = 0;
        p.pid = pid_counter++;

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

    return curr_data;
}

// Makes the list for our simulation later, required starting data
// Step 1: Check interactions between Particles, based on README link
// Step 2: Update list.
// Step 3: Sort list, lowest first
std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
                                   double box_length, double radius, int pnum){
    /* passing curr_data as const reference to avoid the copy and accidental
    overwriting */
    std::vector<Interaction> list(pnum), walltime(6);
    Interaction test;
    double del_x, del_y, del_z, del_vx, del_vy, del_vz, r_tot, rad_d, del_vtot;
    double vdotr;

    // Step 1 -- find interactions
    for (int i = 0; i < pnum; i++){

        for (int k = 0; k < 6; k++ ){
            // setting arbitrarily high... 
            walltime[k].rtime = std::numeric_limits<double>::infinity();
            walltime[k].part1 = i;
            walltime[k].part2 = -k - 1;
        }

        // checking for interactions with the wall.
        if (curr_data[i].vel_x > 0){
            walltime[0].rtime = (box_length - curr_data[i].pos_x) 
                                / curr_data[i].vel_x;
            walltime[0].part2 = -1;
        }

        if (curr_data[i].vel_x < 0){
            walltime[1].rtime = - curr_data[i].pos_x / curr_data[i].vel_x;
            walltime[1].part2 = -2;
        }

        if (curr_data[i].vel_y > 0){
            walltime[2].rtime = (box_length - curr_data[i].pos_y) 
                                / curr_data[i].vel_y;
            walltime[2].part2 = -3;
        }

        if (curr_data[i].vel_y < 0){
            walltime[3].rtime = - curr_data[i].pos_y / curr_data[i].vel_y;
            walltime[3].part2 = -4;
        }

        if (curr_data[i].vel_z > 0){
            walltime[4].rtime = (box_length - curr_data[i].pos_z) 
                                / curr_data[i].vel_z;
            walltime[4].part2 = -5;
        }

        if (curr_data[i].vel_z < 0){
            walltime[5].rtime = - curr_data[i].pos_z / curr_data[i].vel_z;
            walltime[5].part2 = -6;
        }

        std::sort(std::begin(walltime), std::end(walltime),
                  [](const Interaction &dum1, const Interaction &dum2)
                     {return dum1.rtime < dum2.rtime;});

        for (int j = 0; j < pnum; j++){
            if (i != j){

                // simple definitions to make things easier later
                del_x = curr_data[i].pos_x - curr_data[j].pos_x;
                del_y = curr_data[i].pos_y - curr_data[j].pos_y;
                del_z = curr_data[i].pos_z - curr_data[j].pos_z;

                del_vx = curr_data[i].vel_x - curr_data[j].vel_x;
                del_vy = curr_data[i].vel_y - curr_data[j].vel_y;
                del_vz = curr_data[i].vel_z - curr_data[j].vel_z;

                r_tot = 2 * radius;

                // change in velocity * change in r
                vdotr = del_vx*del_x + del_vy*del_y + del_vz*del_z;
                del_vtot = del_vx*del_vx + del_vy*del_vy + del_vz*del_vz;

                // This is under the radical in quad eq., thus "rad_d"
                rad_d = (vdotr * vdotr) -  del_vtot
                        * (del_x*del_x + del_y*del_y + del_z*del_z 
                           - r_tot*r_tot);

                double taptime;
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
                    taptime < list[i].rtime){

                    std::cout << "found one!" << std::endl;
                    test.rtime = taptime;
                    test.part1 = i; //curr_data[i].pid;
                    test.part2 = curr_data[j].pid;
                    list[i] = test;
                }
            }

        }
        if (walltime[0].rtime < list[i].rtime || list[i].rtime == 0){
            list[i] = walltime[0];
        }
        std::cout << '\n' ;
        std::cout << list[i].rtime << '\t' << list[i].part1 << '\t' << i << '\t'
                  << list[i].part2 << '\t' << walltime[0].rtime << '\t'
                  << walltime[0].part1 << '\t' << walltime[0].part2 <<'\n';

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
// UNCHECKED -- CERTAINLY BUG
void simulate(std::vector<Interaction> &interactions, 
              std::vector<Particle> &curr_data, 
              double radius, double mass, double box_length, pnum){

    // opens file for writing
    std::ofstream output("out.dat", std::ofstream::out);

    double del_x, del_y, del_z, J_x, J_y, J_z, J_tot, rtot;
    double del_vx, del_vy, del_vz, del_vtot;
    int count = 0;

    // Note that these are all defined in the material linked in README
    // Step 1
    double final_time = interactions.back().rtime;
    for (double simtime = 0; simtime < final_time; 
        simtime += interactions[0].rtime){

        // changing the 
        for ( int i = 0; i < pnum; i++){
            curr_data[i].pos_x += curr_data[i].vel_x * simtime;
            curr_data[i].pos_y += curr_data[i].vel_y * simtime;
            curr_data[i].pos_z += curr_data[i].vel_z * simtime;

            output << curr_data[i].pid << '\t' << simtime << '\t' 
                   << curr_data[i].pos_x << '\t' << curr_data[i].pos_y
                   << curr_data[i].pos_y << '\t' << curr_data[i].vel_x
                   << curr_data[i].vel_y << '\t' << curr_data[i].vel_z
                   << '\n';

        }

        output << '\n' << '\n';

        if (interactions[0].rtime > 0){
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
    
    
            rtot = sqrt(del_x*del_x + del_y*del_y + del_z*del_z);
            del_vtot = sqrt(del_vx*del_vx + del_vy*del_vy + del_vz*del_vz);
    
            // Step 2
            J_tot = (2 * mass*mass* del_vtot * rtot) / (2*radius * (2 * mass));
    
            J_x = J_tot * del_x / (2 * radius);
            J_y = J_tot * del_y / (2 * radius);
            J_z = J_tot * del_z / (2 * radius);
    
            curr_data[interactions[0].part1].vel_x += J_x / mass;
            curr_data[interactions[0].part1].vel_y += J_y / mass;
            curr_data[interactions[0].part1].vel_z += J_z / mass;
    
            curr_data[interactions[0].part2].vel_x -= J_x / mass;
            curr_data[interactions[0].part2].vel_y -= J_y / mass;
            curr_data[interactions[0].part2].vel_z -= J_z / mass;
        }

        if (interactions[0].rtime < 0){
            switch(interactions[0].part2){
                case -1:
                    curr_data[interactions[0].part1].pos_x -= box_length;
                
                case -2:
                    curr_data[interactions[0].part1].pos_x += box_length;
                
                case -3:
                    curr_data[interactions[0].part1].pos_y -= box_length;
                
                case -4:
                    curr_data[interactions[0].part1].pos_y += box_length;
                
                case -5:
                    curr_data[interactions[0].part1].pos_z -= box_length;
                
                case -6:
                    curr_data[interactions[0].part1].pos_z += box_length;
                
            }

        
        }


        // Step 3 -- update list; TODO
        // UNCHECKED
        interactions = make_list(curr_data, box_length, radius, pnum);
 

        for (const auto &it : interactions){
            std::cout << it.rtime << std::endl;
        }

        std::cout << '\n' << '\n';

        std::cout << simtime << '\t' << count << '\n';
        count++;

    }

    output.close();

}

// Update list during simulate
// NOT FINISHED
std::vector<Interaction> update_list(const std::vector<Particle> &curr_data,
                                     double box_length, double radius,
                                     std::vector<Interaction> list){

    return list;
}
