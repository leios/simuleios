/*------------Barnes_hut.cpp--------------------------------------------------//
*
* Purpose: We start with an octree, then we move onto an N-body simulator
*
*   Notes: We need a separate particle struct to hold pos, mass, radius
*          Depth-first-search
*          force integration
*          visualization
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <cstdio> // For printf (which should replace iostream)
#include <vector>
#include <array>
#include <random>

/*----------------------------------------------------------------------------//
* STRUCTS
*-----------------------------------------------------------------------------*/

// Struct for positions of particles and octree nodes
struct pos{
    double x, y, z;
    pos() : x(0.0), y(0.0), z(0.0) {}
    pos(double x0, double y0, double z0) : x(x0), y(y0), z(z0) {}
};

// Struct for Center of mass
struct center_of_mass{
    pos p;
    double mass;
};

// struct for octree nodes
struct node {
    node *parent;

    // Position of node / box
    pos p;
    double box_length;
    center_of_mass com;

    // Positions of all particles in box
    std::vector<pos> p_pos;
 
    // Nodes are ordered in the following way:
    // Split cube into north / south, then into NSEW again,
    // 0: NNE, 1: NSE, 2: NSW, 3: NNW, 4: SNE, 5: SSE, 6: SSW, 7: SNW
    std::array<node *, 8> children;

    node() : p(pos()), box_length(1.0) {}
    node(pos loc, double length, node *par) : 
        p(loc), box_length(length), parent(par) {}

};

// Function to create random distribution of particles for Octree
std::vector<pos> create_rand_dist(double box_length, int pnum);

// Function to create octree from position data
// Initially, read in root node
void divide_octree(std::vector<pos> &p_pos, node *curr, int box_threshold,
                   double mass);

// Function to check whether a position is within a box
bool in_box(node *curr, pos p);

// Function to create 8 children node for octree
void make_octchild(node *curr);

// Function to find center of mass
// NOTE: some objects may have different masses, maybe read in center of masses
center_of_mass find_com(std::vector<pos> &p_pos, double mass);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    std::vector<pos> p_pos = create_rand_dist(1.0, 10);
    //std::vector<pos> p_pos(1);
    //p_pos[0].x = 0.25; p_pos[0].y = 0.25; p_pos[0].z = 0.25;

    std::cout << '\n' << '\n';

    // Creating root node
    node *root = new node();
    divide_octree(p_pos, root, 1, 0.1);

/*
    for (auto &p : p_pos){
        if (in_box(root, p)){
            std::cout << "found particle" << '\n';
        }
    }
*/

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to create random distribution of particles for Octree
std::vector<pos> create_rand_dist(double box_length, int pnum){

    // Creating vector for particle positions (p_pos)
    std::vector<pos> p_pos;
    p_pos.reserve(pnum);

    // Creating random device to place particles later
    static std::random_device rd;
    int seed = rd();
    static std::mt19937 gen(seed);
    std::uniform_real_distribution<double> 
        box_dist(-box_length * 0.5, box_length * 0.5);

    std::cout << "creating random positions for all " << pnum
              << " particles!" << '\n';
    // Iterating over all particles to create random positions
    for (int i = 0; i < pnum; ++i){
        p_pos.emplace_back(box_dist(gen), box_dist(gen), box_dist(gen));
        printf("%lf %lf %lf \n", p_pos[i].x, p_pos[i].y, p_pos[i].z);
    }
    
    return p_pos;
}

// Function to create octree from position data
// Initially, read in root node
void divide_octree(std::vector<pos> &p_pos, node *curr, int box_threshold, 
                   double mass){
    // Divide node into 8 subnodes (boxes) to work with
    make_octchild(curr);
    
    // Iterating through all the children
    for (auto &child : curr->children){
        // Determine number of particles in our current box
        for (auto &p : p_pos){
            if (in_box(child, p)){
                child->p_pos.push_back(p);
                child->com.p.x += p.x;
                child->com.p.y += p.y;
                child->com.p.z += p.z;
                child->com.mass += 1;
                //child->com.mass +=p.mass;
            }
        }
        child->parent = curr;
        child->com.p.x /= child->com.mass;
        child->com.p.y /= child->com.mass;
        child->com.p.y /= child->com.mass;
        //std::cout << child->p_pos.size() << '\n';
        if (child->p_pos.size() > box_threshold){
            //child->com = find_com(child->p_pos, mass);
            std::cout << child->com.p.x << '\t' << child->com.p.y << '\t'
                      << child->com.mass << '\n';

            divide_octree(child->p_pos, child, box_threshold, mass);
        }
        else if(child->p_pos.size() <= box_threshold && 
                child->p_pos.size() != 0){
            //child->com = find_com(child->p_pos, mass);
            std::cout << child->com.p.x << '\t' << child->com.p.y << '\t'
                      << child->com.mass << '\n';
            std::cout << "Found particle!" << '\n';
        }
        //else{
        //    std::cout << "No particle in box!" << '\n';
        //}

    }

}

// Function to check whether a position is within a box
bool in_box(node *curr, pos p){
    return (p.x >= curr->p.x - curr->box_length * 0.5 && 
            p.x < curr->p.x + curr->box_length * 0.5 &&
            p.y >= curr->p.y - curr->box_length * 0.5 &&
            p.y < curr->p.y + curr->box_length * 0.5 &&
            p.z >= curr->p.z - curr->box_length * 0.5 &&
            p.z < curr->p.z + curr->box_length * 0.5 );
}

// Function to create 8 children node for octree
void make_octchild(node *curr){
    double node_length = curr->box_length * 0.5;

    // iterating through possible locations for new children nodes
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
center_of_mass find_com(std::vector<pos> &p_pos, double mass){
    center_of_mass com;
    for (auto &p : p_pos){
        com.p.x += p.x;
        com.p.y += p.y;
        com.p.z += p.z;
    }

    // Assuming that all particles are of the same weight, this should be fine.
    com.p.x /= p_pos.size();
    com.p.y /= p_pos.size();
    com.p.z /= p_pos.size();

    com.mass = p_pos.size() * mass;

    return com;
}

