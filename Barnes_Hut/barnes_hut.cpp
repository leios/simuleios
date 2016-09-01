/*------------Barnes_hut.cpp--------------------------------------------------//
*
* Purpose: We start with an octree, then we move onto an N-body simulator
*
*   Notes: implement theta
*          force integration
*          visualization -- use OGL, blender, or VTK?
*          to plot use:
*              gnuplot cube_plot.gp -
*
*-----------------------------------------------------------------------------*/

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

const double PARTICLE_MASS = 1;

// Struct for Center of mass
struct particle{
    vec p;
    double mass, radius;

    particle() : mass(PARTICLE_MASS) {}
    explicit particle(vec p, double m = PARTICLE_MASS)
        : p(p), mass(m) {}
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
             com(vec(),0.0) {}
    node(vec loc, double length, node *par) : 
        p(loc), box_length(length), parent(par), children{nullptr}, 
        com(vec(),0.0) {}

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

// Fucntion to output all particle positions
// This function will be used later to plot out what particles actually see
// witht he barnes hut algorithm for gravitational / electrostatic forces
void particle_output(node *curr, std::ofstream &p_output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    std::vector<particle> p_vec = create_rand_dist(1.0, 100);
    //std::vector<particle> p_vec(1);
    //p_vec[0].x = 0.25; p_vec[0].y = 0.25; p_vec[0].z = 0.25;

    std::cout << '\n' << '\n';

    // Creating root node
    node *root = new node();
    divide_octree(p_vec, root, 1, 0.1);
    depth_first_search(root);

    // Defining file for output
    std::ofstream output("out.dat", std::ofstream::out);
    std::ofstream p_output("pout.dat", std::ofstream::out);

/*
    for (auto &p : p_vec){
        if (in_box(root, p)){
            std::cout << "found particle" << '\n';
        }
    }
*/

    octree_output(root, output);
    particle_output(root, p_output);

    output.close();

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to create random distribution of particles for Octree
std::vector<particle> create_rand_dist(double box_length, int pnum){

    // Creating vector for particle vecitions (p_vec)
    std::vector<particle> p_vec;
    p_vec.reserve(pnum);

    // Creating random device to place particles later
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);
    std::uniform_real_distribution<double> 
        box_dist(-box_length * 0.5, box_length * 0.5);

    std::cout << "creating random vecitions for all " << pnum
              << " particles!" << '\n';
    // Iterating over all particles to create random vecitions
    for (int i = 0; i < pnum; ++i){
        p_vec.emplace_back(vec(box_dist(gen), box_dist(gen), box_dist(gen)), 
                           PARTICLE_MASS);
        printf("%lf %lf %lf \n", p_vec[i].p.x, p_vec[i].p.y, p_vec[i].p.z);
    }
    
    return p_vec;
}

// Function to create octree from vecition data
// Initially, read in root node
void divide_octree(std::vector<particle> &p_vec, node *curr,
                   size_t box_threshold, double mass){

    // Divide node into 8 subnodes (boxes) to work with
    make_octchild(curr);
    
    // Iterating through all the children
    for (auto &child : curr->children){
        // Determine number of particles in our current box
        for (auto &p : p_vec){
            if (in_box(child, p)){
                child->p_vec.push_back(p);
                child->com.p += p.p;
                child->com.mass += p.mass;
            }
        }
        child->parent = curr;
        child->com.p /= child->com.mass;
        //std::cout << child->p_vec.size() << '\n';
        if (child->p_vec.size() > box_threshold){
            //child->com = find_com(child->p_vec, mass);
            std::cout << child->com.p.x << '\t' << child->com.p.y << '\t'
                      << child->com.p.z << '\t' << child->com.mass << '\n';

            divide_octree(child->p_vec, child, box_threshold, mass);
        }
        else if(child->p_vec.size() <= box_threshold && 
                child->p_vec.size() != 0){
            //child->com = find_com(child->p_vec, mass);
            std::cout << child->com.p.x << '\t' << child->com.p.y << '\t'
                      << child->com.p.z << '\t' << child->com.mass << '\n';
            std::cout << "Found particle!" << '\n';
        }
        //else{
        //    std::cout << "No particle in box!" << '\n';
        //}

    }

}

// Function to check whether a vecition is within a box
bool in_box(node *curr, particle p){
    return (p.p.x >= curr->p.x - curr->box_length * 0.5 && 
            p.p.x < curr->p.x + curr->box_length * 0.5 &&
            p.p.y >= curr->p.y - curr->box_length * 0.5 &&
            p.p.y < curr->p.y + curr->box_length * 0.5 &&
            p.p.z >= curr->p.z - curr->box_length * 0.5 &&
            p.p.z < curr->p.z + curr->box_length * 0.5 );
}

// Function to create 8 children node for octree
void make_octchild(node *curr){
    double node_length = curr->box_length * 0.5;

    // iterating through vecsible locations for new children nodes
    // This was written by laurensbl
    for (int k = -1; k <= 1; k+= 2){
        for (int j = -1; j <= 1; j += 2){
            for (int i = -1; i <= 1; i +=2){
                int n = 2 * k + j + (i+1)/2 + 3;
                curr->children[n] = new node();
                //std::cout << "n is: " << n << '\n';
                curr->children[n]->box_length = node_length;
                curr->children[n]->p.z = curr->p.z + k * curr->box_length*0.25;
                curr->children[n]->p.y = curr->p.y + j * curr->box_length*0.25;
                curr->children[n]->p.x = curr->p.x + i * curr->box_length*0.25;
            }
        }
    }
}

// Function to find center of mass
particle find_com(std::vector<particle> &p_vec, double mass){
    particle com;
    for (auto &p : p_vec){
        com.p.x += p.p.x;
        com.p.y += p.p.y;
        com.p.z += p.p.z;
    }

    // Assuming that all particles are of the same weight, this should be fine.
    com.p.x /= p_vec.size();
    com.p.y /= p_vec.size();
    com.p.z /= p_vec.size();

    com.mass = p_vec.size() * mass;

    return com;
}

// Function to perform a depth first search of octree
void depth_first_search(node *curr){
    if (!curr){
        return;
    }

    std::cout << curr->p.x << '\t' << curr->p.y << '\t' << curr->p.z << '\n';

    for (auto child : curr->children){
        depth_first_search(child);
    }
}


// Function to output vertex positions of cube(s)
void octree_output(node *curr, std::ofstream &output){

    if (!curr){
        return;
    }

    // Outputting current node vertices

    // find the lower left vertex
    vec llv = vec(curr->p.x - curr->box_length * 0.5,
                  curr->p.y - curr->box_length * 0.5,
                  curr->p.z - curr->box_length * 0.5);

    output << llv.x << '\t' << llv.y << '\t' << llv.z << '\n';
    output << llv.x << '\t' << llv.y << '\t' 
           << llv.z + curr->box_length << '\n';
    output << llv.x << '\t' << llv.y + curr->box_length << '\t' 
           << llv.z + curr->box_length << '\n';
    output << llv.x << '\t' << llv.y + curr->box_length << '\t' 
           << llv.z << '\n';
    output << llv.x << '\t' << llv.y << '\t' << llv.z << '\n';
    output << '\n' << '\n';

    output << llv.x + curr->box_length << '\t' << llv.y << '\t' 
           << llv.z << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y << '\t' 
           << llv.z + curr->box_length << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y + curr->box_length 
           << '\t' << llv.z +curr->box_length << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y + curr->box_length 
           << '\t' << llv.z << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y << '\t' 
           << llv.z << '\n';
    output << '\n' << '\n';

    output << llv.x << '\t' << llv.y << '\t' << llv.z << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y << '\t' 
           << llv.z << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y + curr->box_length 
           << '\t' << llv.z << '\n';
    output << llv.x << '\t' << llv.y + curr->box_length << '\t' 
           << llv.z << '\n';
    output << llv.x << '\t' << llv.y << '\t' << llv.z << '\n';
    output << '\n' << '\n';

    output << llv.x << '\t' << llv.y << '\t' 
           << llv.z + curr->box_length << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y << '\t' 
           << llv.z + curr->box_length << '\n';
    output << llv.x + curr->box_length << '\t' << llv.y + curr->box_length 
           << '\t' << llv.z + curr->box_length << '\n';
    output << llv.x << '\t' << llv.y + curr->box_length << '\t' 
           << llv.z + curr->box_length << '\n';
    output << llv.x << '\t' << llv.y << '\t' 
           << llv.z + curr->box_length << '\n';
    output << '\n' << '\n';

    // Recursively outputing internal boxes
    for (auto child : curr->children){
        octree_output(child, output);
    }
 
}

// Function to output particle positions
void particle_output(node *curr, std::ofstream &p_output){

    if (!curr){
        return;
    }

    if (curr->p_vec.size() == 1){
        p_output << curr->com.p.x << '\t' << curr->com.p.y << '\t' 
                 << curr->com.p.z << '\n';
    }

    // Recursively outputting additional particles
    for (auto child : curr->children){
        particle_output(child, p_output);
    }

}

