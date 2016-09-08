/*------------Barnes_hut.cpp--------------------------------------------------//
*
* Purpose: We start with an octree, then we move onto an N-body simulator
*
*   Notes: think about regenerating octree every timestep, happylittlerat: 
*              http://www.randygaul.net/2013/08/06/dynamic-aabb-tree/ 
*          force integration -- All particles drift to a plane
*          visualization -- use OGL?
*          to plot use:
*              gnuplot cube_plot.gp -
*          we may have to ignore particles that are too far away
*
*-----------------------------------------------------------------------------*/

#include "barnes_hut.h"

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    int numsteps = 1000;
    // Defining file for output
    std::vector <std::ofstream> octree_files;
    std::string filename;
    for (int i = 0; i < numsteps; i++){
        filename = "octree" + std::to_string(i) + ".dat";
        octree_files.emplace_back(filename, std::ofstream::out);
    }
    std::ofstream p_output("pout.dat", std::ofstream::out);

    // Creating the octree
    //std::vector<particle> p_vec = create_rand_dist(1, 1);
    double vel = sqrt(10);
    double dt = 0.01;
    std::vector<particle> p_vec;
    p_vec.reserve(1);
    p_vec.emplace_back(vec(1, 0, 0), 
                       vec(0, vel, 0), vec(), PARTICLE_MASS);
    p_vec[0].prev_p = vec(cos(vel* dt), sin(vel * dt),0);
    print(p_vec[0].prev_p);

    node *root = make_octree(p_vec, 1000);
    divide_octree(root, 1);

    for (int i = 0; i < numsteps; ++i){
        node *root = make_octree(p_vec, 1000);
        divide_octree(root, 1);
        if (i % 100 || i == 0){
            particle_output(root, p_output);
            p_output << "\n\n";
        }
        force_integrate(root, 0.01);
        traverse_post_order(root, [](node* node){
            delete node;
        });
        //octree_output(root, octree_files[i]);
    }

/*
    // Show all particle positions
    particle_output(root, std::cout);
    particle_output(root, p_output);

    std::cout << "\n\n";
    p_output << "\n\n";

    force_integrate(root, 0.001);
    force_integrate(root, 0.001);

    particle_output(root, std::cout);
    particle_output(root, p_output);
    octree_output(root, output);
*/

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to create random distribution of particles for Octree
std::vector<particle> create_rand_dist(double box_length, int pnum){
    // Creating vector for particle positions (p_vec)
    std::vector<particle> p_vec;
    p_vec.reserve(pnum);

    // Creating random device to place particles later
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> 
        box_dist(-box_length * 0.5, box_length * 0.5);

    // Iterating over all particles to create random vecitions
    for (int i = 0; i < pnum; ++i){
        p_vec.emplace_back(vec(box_dist(gen), box_dist(gen), box_dist(gen)), 
                           vec(), vec(), PARTICLE_MASS);
    }
    
    return p_vec;
}

node* make_octree(std::vector<particle> &p_vec, double box_length) {
    auto root = new node(vec(), box_length, nullptr);
    root->p_vec.reserve(p_vec.size());

    for (auto& p : p_vec) {
        root->p_vec.push_back(&p);
    }
    return root;
}

// Function to create octree from vecition data
// Initially, read in root node
void divide_octree(node *curr, size_t box_threshold){

    // Divide node into 8 subnodes (boxes) to work with
    make_octchild(curr);
    
    // Iterating through all the children
    for (auto &child : curr->children){
        // Determine number of particles in our current box
        for (auto p : curr->p_vec){
            if (in_box(child, p)){
                child->p_vec.push_back(p);
                child->com.p += p->p * p->mass;
                child->com.mass += p->mass;
            }
        }
        child->com.p /= child->com.mass;
        if (child->p_vec.size() > box_threshold){
            divide_octree(child, box_threshold);
        }
    }
}

// Function to check whether a vecition is within a box
bool in_box(node *curr, particle *p){
    double half_box = curr->box_length * 0.5;
    return p->p.x >= curr->p.x - half_box && 
           p->p.x <  curr->p.x + half_box &&
           p->p.y >= curr->p.y - half_box &&
           p->p.y <  curr->p.y + half_box &&
           p->p.z >= curr->p.z - half_box &&
           p->p.z <  curr->p.z + half_box ;
}

// Function to create 8 children node for octree
void make_octchild(node *curr){
    double node_length = curr->box_length * 0.5;
    double quarter_box = curr->box_length * 0.25;

    // iterating through vecsible locations for new children nodes
    // This was written by laurensbl
    for (int k = -1; k <= 1; k+= 2){
        for (int j = -1; j <= 1; j += 2){
            for (int i = -1; i <= 1; i +=2){
                int n = 2 * k + j + (i+1)/2 + 3;
                curr->children[n] = new node();
                curr->children[n]->parent = curr;
                curr->children[n]->box_length = node_length;
                curr->children[n]->p.z = curr->p.z + k * quarter_box;
                curr->children[n]->p.y = curr->p.y + j * quarter_box;
                curr->children[n]->p.x = curr->p.x + i * quarter_box;
            }
        }
    }
}

// Function to perform a depth first search of octree
void depth_first_search(node *curr){
    if (!curr){
        return;
    }

    for (auto child : curr->children){
        depth_first_search(child);
    }
}


// Function to output vertex positions of cube(s)
void octree_output(node *curr, std::ostream &output){
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
void particle_output(node *curr, std::ostream &p_output){
    if (!curr){
        return;
    }

    if (curr->p_vec.size() == 1){
        p_output << curr->p_vec[0]->p.x << '\t' << curr->p_vec[0]->p.y << '\t' 
                 << curr->p_vec[0]->p.z << '\n';
    }

    // Recursively outputting additional particles
    for (auto child : curr->children){
        particle_output(child, p_output);
    }

}

// Function to find acceleration of particle in Barnes Hut tree
void force_integrate(node *root, double dt){

    // Going through all particles in p_vec, finding the new acceleration
    #pragma omp parallel for
    for (size_t i = 0; i < root->p_vec.size(); i++){
        root->p_vec[i]->acc = vec();
        RKsearch(root, root->p_vec[i]);
    }   

    // Going through all the particles and updating position 
    #pragma omp parallel for
    for (size_t i = 0; i < root->p_vec.size(); i++){
        //RK4(root->p_vec[i], dt);
        verlet(root->p_vec[i], dt);
    }

}

// Recursive function to find acceleration of particle in tree
void RKsearch(node *curr, particle *part){

    vec d = -part->p;
    double inverse_r = 1/length(d);

    if (inverse_r > 10){
        inverse_r = 10;
    }
    if (inverse_r < -10){
        inverse_r = -10;
    }


    //part->acc += d*(G * 1E11 * inverse_r * inverse_r * inverse_r);
    part->acc += d * inverse_r * 10;
    //print(part->acc);
    //print(d);

/*
    if (!curr){
        return;
    }
    if (curr->p_vec.size() == 1 && curr->p_vec[0] == part){
        return;
    }

    // Defining a few variables, distance and inverse_r (save those divisions)
    vec d = curr->com.p - part->p;
    double inverse_r = 1/length(d);

    // Defining new theta of current node
    double theta_2 = curr->box_length * inverse_r;

    // find the new acceleration due to the current node

    if (theta_2 <= THETA){
        //double acc = G * curr.com.mass * inverse_r * inverse_r;
        // a = GM/r^2 * norm(r)
        part->acc += d*(G * curr->com.mass * inverse_r * inverse_r * inverse_r);
    }
    // if we are in a leaf node
    else if (curr->p_vec.size() == 1){
        part->acc += d*(G * curr->com.mass * inverse_r * inverse_r * inverse_r);
    }
    // if above thresh THETA, then search again.
    else{
        for (auto child : curr->children){
            RKsearch(child, part);
        }
    }
    
*/
}

// Actual implementation of Runge-Kutta 4 method
// Note: It's assumed that all particles have already updated their acc.
void RK4(particle *part, double dt){
    // For RK4, we need an array of size 4 for acc, vel, and pos
    vec pos[4], vel[4], acc[4];

    // Populating the arrays
    pos[0] = part->p;
    vel[0] = part->vel;
    acc[0] = part->acc;

    pos[1] = pos[0] + 0.5 * dt * vel[0];
    vel[1] = vel[0] + 0.5 * dt * acc[0];
    acc[1] = part->acc;

    pos[2] = pos[0] + 0.5 * vel[1] * dt;
    vel[2] = vel[0] + 0.5 * acc[1] * dt;
    acc[2] = part->acc;

    pos[3] = pos[0] + vel[2] * dt;
    vel[3] = vel[0] + acc[2] * dt;
    acc[3] = part->acc;

    part->p = pos[0] + (dt / 6) * (vel[0] + 2 * vel[1] + 2 * vel[2] + vel[3]);
    part->vel = vel[0] + (dt / 6) * (acc[0] + 2 * acc[1] + 2 * acc[2] + acc[3]);
}

// Simple implementation of verlet algorithm as a test against RK4
void verlet(particle *part, double dt){
    vec temp = part->p;
    part->p = 2 * part->p - part->prev_p + part->acc * dt*dt;
    part->prev_p = temp;   
}
