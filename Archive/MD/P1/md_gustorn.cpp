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

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "math.h"

namespace sim {
    /* putting our stuff in its own namespace so we can call "time" "time"
    without colliding with names from the standard library */
    using time = double;

/*----------------------------------------------------------------------------//
* STRUCTS AND FUNCTIONS
*-----------------------------------------------------------------------------*/

// Holds our data in a central struct, to be called mainly in a vector
struct Particle {
    Point pos;
    Vector vel;
    uint64_t pid;
    time ts;
};

// holds interaction data
struct Interaction {
    time rtime;
    uint64_t first, second;

    Interaction() {}
    Interaction(time t, uint64_t p1, uint64_t p2)
        : rtime{t}, first{p1}, second{p2} {}
};

inline bool operator<(const Interaction& lhs, const Interaction& rhs) {
    return lhs.rtime < rhs.rtime;
}

// Makes starting data
std::vector<Particle> populate(uint64_t pnum, double box_length, double max_vel);

// Makes the interactions for our simulation later, required starting data
void update_interactions(const std::vector<Particle>& particles,
                         std::vector<Interaction>& interactions,
                         double box_length, double radius);

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
void simulate(std::vector<Interaction>& interactions,
              std::vector<Particle>& particles,
              double radius, double mass, double box_length);


/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Makes starting data
std::vector<Particle> populate(uint64_t pnum, double box_length, double max_vel){
    std::vector<Particle> particles(pnum);

    uint64_t pid_counter = 0u;
    for (auto &p : particles){ //read: for all particles p in particles
        p.ts = 0;
        p.pid = pid_counter++;
        p.pos = uniform_in_bounds2d<Point>(0, box_length);
        p.vel = uniform_in_bounds2d<Vector>(0, max_vel);
    }

    return particles;
}

// Makes the interactions for our simulation later, required starting data
// Step 1: Check interactions between Particles, based on README link
// Step 2: Update interactions.
// Step 3: Sort interactions, lowest first
void update_interactions(const std::vector<Particle>& particles,
                         std::vector<Interaction>& interactions,
                         double, double radius) {
    // Step 1 -- find interactions
    for (auto i = 0u; i < particles.size() - 1; ++i) {
        for (auto j = i + 1; j < particles.size(); ++j) {
            auto& first = particles[i];
            auto& second = particles[j];

            Vector del_p = first.pos - second.pos;
            Vector del_v = first.vel - second.vel;

            double sigma = 2 * radius;

            // This is under the radical in quad eq., thus "rad_d"
            double rad_d = (square(dot(del_v, del_v))
                           - 4 * dot(del_v, del_v)
                           * (dot(del_p, del_p) - square(sigma)));

            if (dot(del_p, del_v) >= 0 || rad_d < 0){
                continue;
            }

            double check = (-dot(del_v, del_p)+ sqrt(rad_d)) / (2 * dot(del_v, del_v));

            std::cout << "found one!" << std::endl;
            interactions.emplace_back(check, i, j);
        }
    }

    // Step 3 -- sort the interactions

    // The std::sort command wants to know 3 things:
    //     1. Where to start sorting
    //     2. Where to stop sorting
    //     3. Which element value is greater
    //
    // To check #3, we use the user-defined "<" operator of Interaction.
    std::sort(std::begin(interactions), std::end(interactions));
}

// This performs our MD simulation with a vector of interactions
// Also writes simulation to file, specified by README
// Note: Time-loop here
// Step 1: loop over timesteps, defined in interactions
// Step 2: model the interaction of particles that interact
// Step 3: update interactions
// Step 4: output positions to file
// UNCHECKED
void simulate(std::vector<Interaction>& interactions,
              std::vector<Particle>& particles,
              double radius, double mass, double box_length) {

    // Note that these are all defined in the material linked in README
    // Step 1
    for (double simtime = 0; simtime < interactions.back().rtime;
         simtime += interactions[0].rtime){
        auto& first = particles[interactions[0].first];
        auto& second = particles[interactions[0].second];

        Vector del_p = first.pos + first.vel * simtime
                       - second.pos + second.vel * simtime;
        Vector del_v = first.vel - second.vel;

        double rtot = del_p.length();
        double del_vtot = del_v.length();

        // Step 2
        double J_tot = (2 * mass * mass * del_vtot * rtot) / (2 * radius * (2 * mass));

        Vector J = J_tot * del_p / (2 * radius);

        first.vel += J / mass;
        second.vel += J / mass;

        // Step 3 -- update interactions; TODO
        // UNCHECKED
        update_interactions(particles, interactions, box_length, radius);
    }
}

}

// C++ doesn't need the explicit void in main()
int main() {
    uint64_t pnum = 10000;
    double box_length = 10, max_vel = 0.01;

    std::vector<sim::Interaction> interactions;
    std::vector<sim::Particle> particles = sim::populate(pnum, box_length, max_vel);
    sim::update_interactions(particles, interactions, box_length, 0.1);

    // Iterating by const& because we do not intend to change to objects
    for (const auto &p : particles){
        std::cout << p.pos.x << std::endl;
    }

    std::cout << std::endl << std::endl;
    for (const auto &it : interactions){
        std::cout << it.rtime << std::endl;
    }

    return 0;
}
