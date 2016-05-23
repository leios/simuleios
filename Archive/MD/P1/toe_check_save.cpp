#include <vector>
#include <cassert>
#include <iostream>

/*
This file is intended to test if make_list and simulate work correctly
by setting up simple collisions and verifying the result.
There are some collisions that are not so easy after all, so their
expected values may not be correct.
You are supposed to call test_all_interactions before the actual
simulation, if everything works as expected nothing is printed, otherwise
the actual and expected outputs are printed to compare.
*/

namespace sim{
    using time = double;
}

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

//allow easy printing of particles
std::ostream &operator << (std::ostream &os, const Particle &p){
    return os << p.pid << ":pos={" << p.pos_x << ", " << p.pos_y << ", "
        << p.pos_z << "} " << "vel={" << p.vel_x << ", " << p.vel_y << ", "
        << p.vel_z << '}';
}

bool test_interaction(SimulationTest testID){
    std::vector<Particle> startState; //starting situation as drawn below
    std::vector<Particle> endState; //state we are supposed to end up in
    sim::time simulationtime; //how long the simulation should take
    int ID = 0;
    //make creating particles less confusing
    auto makeParticle = [&ID](std::pair<double, double> pos,
        std::pair<double, double> vel){
        return Particle{ ID++, 0, pos.first, pos.second, 1, vel.first,
            vel.second, 0 };
    };
    //check for invalid cases
    assert(testID != SimulationTest::end);
    assert(int(testID) >= int(SimulationTest::singleDirectCollision) &&
        int(testID) < int(SimulationTest::end));

    switch (testID){
        case SimulationTest::singleDirectCollision:
            /* single simple direct collision
            0:    *->  <-*
            1:     <-**->
            2:  <-*      *->
            */
            startState.emplace_back(makeParticle({ 1, 1 }, { 1, 0 }));
            startState.emplace_back(makeParticle({ 3, 1 }, { -1, 0 }));
            ID = 0;
            endState.emplace_back(makeParticle({ 1, 1 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 3, 1 }, { 1, 0 }));
            simulationtime = 2;
            break;
        case SimulationTest::doubleDirectCollision:
            /* 2 simple direct collisions at the same time
            0:   *->  <-*
                 *->  <-*
            1:    <-**->
                  <-**->
            2: <-*      *->
               <-*      *->
            */
            startState.emplace_back(makeParticle({ 1, 1 }, { 1, 0 }));
            startState.emplace_back(makeParticle({ 3, 1 }, { -1, 0 }));
            startState.emplace_back(makeParticle({ 1, 2 }, { 1, 0 }));
            startState.emplace_back(makeParticle({ 3, 2 }, { -1, 0 }));
            ID = 0;
            endState.emplace_back(makeParticle({ 1, 1 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 3, 1 }, { 1, 0 }));
            endState.emplace_back(makeParticle({ 1, 2 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 3, 2 }, { 1, 0 }));
            simulationtime = 2;
            break;
        case SimulationTest::angledCollision:
            /*
            0:     *
                   |
                   v
               *-->



            1:

                *-->
               *
               |
               v



            2:

                    *-->


               *
               |
               v
            */
            startState.emplace_back(makeParticle({ 3, 1 }, { 0, 1 }));
            startState.emplace_back(makeParticle({ 1, 3 }, { 1, 0 }));
            ID = 0;
            //this one is difficult to calculate... We wrote a program for that
            endState.emplace_back(makeParticle({ 3, 1 }, { 0, 1 })); //TODO
            endState.emplace_back(makeParticle({ 1, 3 }, { 1, 0 })); //TODO
            simulationtime = 2; //TODO
            break;
        case SimulationTest::multiCollision:
            /*
            0:    *->  *->  *->  <-*  <-*  <-*
            3:       *->  *-><-**-><-*  <-*
            5:         *-><-**-><-**-><-*
            7:         <-**-><-**-><-**->
            9:       <-*  <-**-><-**->  *->
            11:     <-*  <-*  <-**->  *->  *->
            13:   <-*  <-*  <-*    *->  *->  *->
            */
            startState.emplace_back(makeParticle({ 1, 1 }, { 1, 0 }));
            startState.emplace_back(makeParticle({ 3, 1 }, { 1, 0 }));
            startState.emplace_back(makeParticle({ 5, 1 }, { 1, 0 }));
            startState.emplace_back(makeParticle({ 7, 1 }, { -1, 0 }));
            startState.emplace_back(makeParticle({ 9, 1 }, { -1, 0 }));
            startState.emplace_back(makeParticle({ 11, 1 }, { -1, 0 }));
            ID = 0;
            endState.emplace_back(makeParticle({ 1, 1 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 3, 1 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 5, 1 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 7, 1 }, { 1, 0 }));
            endState.emplace_back(makeParticle({ 9, 1 }, { 1, 0 }));
            endState.emplace_back(makeParticle({ 11, 1 }, { 1, 0 }));
            simulationtime = 6; //not completely sure it takes 6 timeunits
            break;
        case SimulationTest::wallbounce:
            /*
            0:   *-> |
            1:    <-*|
            2: <-*   |
            */
            startState.emplace_back(makeParticle({ 1, 1 }, { -1, 0 }));
            ID = 0;
            endState.emplace_back(makeParticle({ 1, 1 }, { 1, 0 }));
            simulationtime = 1;
            break;
        case SimulationTest::speedCollision:
            /*
            0:  *-->   <-*
            1:    *--><-*
            2:       **--->     ???
            3:       *   *--->  ???
            */
            startState.emplace_back(makeParticle({ 1, 1 }, { 2, 0 }));
            startState.emplace_back(makeParticle({ 3, 1 }, { -1, 0 }));
            ID = 0;
            //Also hard to figure out from just looking at it... Physician help
            endState.emplace_back(makeParticle({ 3, 1 }, { -1, 0 }));
            endState.emplace_back(makeParticle({ 3, 1 }, { -1, 0 }));
            simulationtime = 1;
            break;
        case SimulationTest::nudge:
            /* Is this how physics works?
            0:  *-->  *->
            1:    *--> *->
            2:      *-->*->
            3:        *--*->
            4:          *-*->
            5:            **-->
            6:             *-*-->
            7:              *->*-->
            8:               *-> *-->
            */
            startState.emplace_back(makeParticle({ 1, 1 }, { 2, 0 }));
            startState.emplace_back(makeParticle({ 3, 1 }, { 1, 0 }));
            ID = 0;
            endState.emplace_back(makeParticle({ 4, 1 }, { 1, 0 }));
            endState.emplace_back(makeParticle({ 6, 1 }, { 2, 0 }));
            simulationtime = 2;
        default:
            assert(false); //forgot a case
    }
    //TODO:
    //make simulate quit when encountering a terminateSimulation Interaction
    Interaction terminateSimulation{ simulationtime, -1, -1 };
    //get interactions
    std::vector<Interaction> interactions = make_list(startState, 100, .5);
    //add termination Interaction
    interactions.push_back(std::move(terminateSimulation));
    //simulate
    simulate(startState, interactions);
    if (startState == endState){
        //TODO: also check if passed time == simulationtime
        return true;
    }
    else{
        std::cout << "\nTest " << int(testID) << " failed\n";
        std::cout << "State is:\n";
        for (const auto &p : startState){
            std::cout << p << ' ';
        }
        std::cout << "\nState should be:\n";
        for (const auto &p : endState){
            std::cout << p << ' ';
        }
        return false;
    }
}

void test_all_interactions(){
    for (int i = 0; i < int(SimulationTest::end); ++i){
        //maybe abort after first failure?
        test_interaction(SimulationTest(i));
    }
}

