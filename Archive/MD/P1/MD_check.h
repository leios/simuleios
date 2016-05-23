/*-------------MD_check.h-----------------------------------------------------//
*
*              MD check -- checking the physical accuracy of our MD sim
*
* Purpose: header for MD_check.cpp
*
*-----------------------------------------------------------------------------*/
#ifndef MD_CHECK_H
#define MD_CHECK_H

#include <vector>
#include <cassert>
#include <iostream>

namespace sim{
    using time = double;
}
/*
//should think about putting stuff in headers
struct Particle{
    int pid;
    sim::time ts;
    double pos_x, pos_y, pos_z, vel_x, vel_y, vel_z;
};

struct Interaction{
    sim::time rtime;
    int part1, part2;
};
*/


std::vector<Interaction> make_list(const std::vector<Particle> &curr_data,
    double box_length, double radius);
void simulate(std::vector<Particle> &particles,
    const std::vector<Interaction> &interactions);
enum class SimulationTest{
    singleDirectCollision = 1,
    doubleDirectCollision,
    angledCollision,
    multiCollision,
    wallbounce,
    speedCollision,
    nudge,
    end
};
bool test_interaction(SimulationTest testID);

#endif MD_CHECK_H
