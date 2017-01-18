/*-------------affinity_prop.cpp----------------------------------------------//
*
* Purpose: Cluster points with affinity propagation
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include "../visualization/cairo/cairo_vis.h"

struct Parameter{
    std::vector<vec> part;
    std::vector<double> resp, avail, sim;
    std::vector<bool> exemp;
};

// Function to create a random distribution of points
Parameter init_parameters(int pnum);

// Function to update responsibilities
void update_responsibilities(Parameter &par);

// Function to update availabilities
void update_availabilities(Parameter &par);

// Function to update exemplar list
void choose_exemplars(Parameter &par);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){

    Parameter par = init_parameters(10);

    int timesteps = 100;

    for (int i = 0; i < timesteps; i++){
        update_responsibilities(par);
        update_availabilities(par);
        choose_exemplars(par);  
    }
    
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function to create a random distribution of points
Parameter init_parameters(int pnum){

    // Note that avail and resp are in principle 2d vectors
    Parameter par;
    par.part.reserve(pnum);
    par.exemp.reserve(pnum);
    par.avail.reserve(pnum*pnum);
    par.resp.reserve(pnum*pnum);
    par.sim.reserve(pnum*pnum);

    // Creating random distribution of particles between 0 and 1
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(0,1);

    // Initializing a random distribution for positions
    vec temp;
    for (int i = 0; i < pnum; i++){
        temp.x = pos_dist(gen);
        temp.y = pos_dist(gen);
        par.part.push_back(temp);
        par.exemp.push_back(0.0);
        
    }

    // Initializing avail and resp to 0. Theres a cleaner way to do this.
    for (int i = 0; i < pnum*pnum; i++){
        par.avail.push_back(0.0);
        par.resp.push_back(0.0);
    }

    int index = 0;
    for (int i = 0; i < pnum; i++){
        for (int j = 0; j < pnum; j++){
            index = i*pnum + j;
            par.sim[index] = distance(par.part[i], par.part[j]);
        }
    }

    return par;

}

// Function to update responsibilities
void update_responsibilities(Parameter &par){

    // two indices, index2 is for finding the maximum
    int index = 0, index2 = 0;

    // double temp necessary to find maximum
    double temp;

    // To find the responsibiliy, we check each individual's desire to be lead 
    // by other points in the simulation
    // resp = sim - best other representative (max(a(i,k') + s(i.k')))
    for (size_t i = 0; i < par.part.size(); i++){
        for (size_t j = 0; j < par.part.size(); j++){
            index = i*par.part.size() + j;
            par.resp[index] = par.sim[index];
            temp = 0;

            // Find the maximum of all other points
            for (size_t k = 0; k < par.part.size(); k++){
                index2 = i*par.part.size() + k;
                if (par.avail[index2] + par.sim[index2] > temp){
                    temp = par.avail[index2] + par.sim[index2];
                }
            }

            par.resp[index] += temp;
        }
    }
}

// Function to update availabilities
void update_availabilities(Parameter &par){

    // Following a similar structure for finding the responsibilities above
    int index = 0, index2 = 0;
    double temp;

    // To find the availability, we check each possible representative's 
    // desire to be work with each individual point
    // avail = self- resp + maximum other resp to any other individual
    //         (min(0, r(k,k) + sum(max(0,resp(i',k))))
    for (size_t i = 0; i < par.part.size(); i++){
        for (size_t j = 0; j < par.part.size(); j++){
            index = par.part.size() * i + j;
            temp = 0;
            par.avail[index] = par.resp[par.part.size() * i + i];

            // Finding the maximum other point
            for (size_t k = 0; k < par.part.size(); k++){
                if (j != k){
                    index2 = k * par.part.size() + j;
                    if (par.resp[index2] < 0 && par.resp[index2] > temp){
                        temp = par.resp[index2];
                    }
                }
            }
            par.avail[index] += temp;

            // setting to 0, if too large;
            if (par.avail[index] > 0){
                par.avail[index] = 0;
            }
            
        }
    }
}

// Function to update exemplar list
void choose_exemplars(Parameter &par){

    double temp;
    int index = 0;
    int choice;

    // going through all individuals to see how good they are at representing
    // the public
    for (size_t i = 0; i < par.part.size(); i++){
        temp = 0;
        for (size_t j = 0; j < par.part.size(); j++){
            index = i*par.part.size() + j;
            if (par.avail[index] + par.resp[index] > temp){
                temp = par.avail[index] + par.resp[index];
                choice = j;
            }
        }

        if (choice == i){
            par.exemp[i] = 1;
        }
        else{
            par.exemp[i] = 0;
        }
        
    }
}
