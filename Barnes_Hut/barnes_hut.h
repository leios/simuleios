/*-------------barnes_hut.h---------------------------------------------------//
*
* Purpose: Header file for barnes_hut.cpp
*
*-----------------------------------------------------------------------------*/

#ifndef BARNES_HUT_H
#define BARNES_HUT_H

#include <iostream>
#include <cstdio> // For printf (which should replace iostream)
#include <vector>
#include <array>
#include <random>
#include <fstream>
#include "vec.h"

/*----------------------------------------------------------------------------//
* STRUCTS
*-----------------------------------------------------------------------------*/

const double PARTICLE_MASS = 1E10;
const double G = 6.67408E-11;
const double THETA = 0.5;

// Struct for Center of mass
struct particle{
    vec p;
    vec vel;
    vec acc;
    double mass, radius;

    particle() : mass(PARTICLE_MASS) {}
    explicit particle(vec p, vec v, vec a, double m = PARTICLE_MASS)
        : p(p), vel(v), acc(a), mass(m) {}
};

// struct for octree nodes
struct node {

    // Position of node / box
    vec p;
    double box_length;
    node *parent;

    // Nodes are ordered in the following way:
    // Split cube into north / south, then into NSEW again,
    // 0: NNE, 1: NSE, 2: NSW, 3: NNW, 4: SNE, 5: SSE, 6: SSW, 7: SNW
    std::array<node *, 8> children;

    particle com;

    // Positions of all particles in box
    std::vector<particle> p_vec;
 
    node() : p(vec()), box_length(1.0), parent(nullptr), children{nullptr},
             com(vec(), vec(), vec(),0.0) {}
    node(vec loc, double length, node *par) : 
        p(loc), box_length(length), parent(par), children{nullptr}, 
        com(vec(), vec(), vec(),0.0) {}

};

// Function to create random distribution of particles for Octree
std::vector<particle> create_rand_dist(double box_length, int pnum);

// Function to create octree from vecition data
// Initially, read in root node
void divide_octree(std::vector<particle> &p_vec, node *curr, 
                   size_t box_threshold, double mass);

// Function to check whether a particle is within a box
bool in_box(node *curr, particle p);

// Function to create 8 children node for octree
void make_octchild(node *curr);

// Function to find center of mass
// NOTE: some objects may have different masses, maybe read in center of masses
particle find_com(std::vector<particle> &p_vec, double mass);

// Function to perform a depth first search of octree
void depth_first_search(node *curr);

// Function to output vertex positions of cube(s)
void octree_output(node *curr, std::ofstream &output);

// Function to output all particle positions
// This function will be used later to plot out what particles actually see
// witht he barnes hut algorithm for gravitational / electrostatic forces
void particle_output(node *curr, std::ofstream &p_output);

// Function to find acceleration of particle in Barnes Hut tree
void force_integrate(node *root, double dt);

// Recursive function to find acceleration of particle in tree
void RKsearch(node *curr, particle &part);

// Actual implementation of Runge-Kutta 4 method
// Note: It's assumed that all particles have already updated their acc.
void RK4(particle &part, double dt);

#endif
