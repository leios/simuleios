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

    particle() : mass(PARTICLE_MASS), radius(0.0) {}
    particle(vec p, vec v, vec a, double m = PARTICLE_MASS)
        : p(p), vel(v), acc(a), mass(m), radius(0.0) {}
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
    std::array<node*, 8> children;

    particle com;

    // Positions of all particles in box
    std::vector<particle*> p_vec;
 
    node() : p(0, 0, 0), box_length(1.0),
             parent(nullptr), children{nullptr},
             com(vec(), vec(), vec(), 0.0) {}

    node(vec loc, double length, node *par) : 
        p(loc), box_length(length),
        parent(par), children{nullptr}, 
        com(vec(), vec(), vec(),0.0) {}

    ~node();
};

// Function to create random distribution of particles for Octree
std::vector<particle> create_rand_dist(double box_length, int pnum);

// Creates the root node for an octree, given a list of particles and maxim
node* make_octree(std::vector<particle> &particles);

// Divides an octree node if 
void divide_octree(node *curr, size_t box_threshold);

// Function to check whether a particle is within a box
bool in_box(node *curr, particle *p);

// Function to create 8 children node for octree
void make_octchild(node *curr);

// Function to perform a depth first search of octree
void depth_first_search(node *curr);

// Function to output vertex positions of cube(s)
void octree_output(node *curr, std::ostream &output);

// Function to output all particle positions
// This function will be used later to plot out what particles actually see
// witht he barnes hut algorithm for gravitational / electrostatic forces
void particle_output(node *curr, std::ostream &p_output);

// Function to find acceleration of particle in Barnes Hut tree
void force_integrate(node *root, double dt);

// Recursive function to find acceleration of particle in tree
void RKsearch(node *curr, particle *part);

// Actual implementation of Runge-Kutta 4 method
// Note: It's assumed that all particles have already updated their acc.
void RK4(particle *part, double dt);

// Traverses the tree in a post-order manner, executing the function given
// as the argument for each node
template <typename T>
void traverse_post_order(node* curr, const T& fn) {
    if (!curr) {
        return;
    }

    for (auto child : curr->children) {
        traverse_post_order(child, fn);
    }

    fn(curr);
}

inline node::~node() {
    traverse_post_order(this, [](node* node) { delete node; });
}

#endif
