/*-------------Anderson.cpp---------------------------------------------------//
*
*              Anderson.cpp -- Diffusion Monte Carlo for Schroedy
*
* Purpose: Implement the quantum Monte Carlo (diffusion) by Anderson:
*             http://www.huy-nguyen.com/wp-content/uploads/QMC-papers/Anderson-JChemPhys-1975.pdf
*          For H3+, with 3 protons and 2 electrons
*
*    Note: This algorithm may be improved by later work
*          A "psip" is an imaginary configuration of electrons in space.
*          Requires c++11 for random and Eigen for matrix
*          Protons assume to be at origin
*          A lot of this algorithm is more easily understood here:
*              http://www.thphys.uni-heidelberg.de/~wetzel/qmc2006/KOSZ96.pdf
*
*-----------------------------------------------------------------------------*/

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>

constexpr size_t DOF = 6;
constexpr size_t MAX_SIZE = 2000;
constexpr size_t INITIAL_SIZE = 1000;
constexpr double TIMESTEP = 0.1;
constexpr double RADIUS = 1.8;

using coord = std::array<double, DOF>;

struct proton_pos{
    double pos[3][3] = {{0}};
};

struct particle {
    coord coords;
    int id;
    int m_n;
    double potential;

    // This is the constructor used for the initial state creation
    particle(int id, const coord& coords)
      : coords(coords), id(id),  m_n(-1), potential(0.0) {}

    // Used in branching to create a replica of an other particle.
    // Sets m_n to 1 so even if this was to be processed, it will
    // be ignored as per the algorithm described in the Kosztin paper
    particle(int id, const particle& p)
      : coords(p.coords), id(id),  m_n(1), potential(p.potential) {}

};

struct h3plus {
    std::vector<particle> particles;
    double v_ref, dt, energy;
    int global_id;

    // This wouldn't be strictly necessary,
    // but pre-allocating should speed up the program a bit
    h3plus(size_t reserved_size) {
        particles.reserve(reserved_size);
    }

    proton_pos proton;

};

// Populate a distribution of particles for QMC
h3plus generate_initial(size_t initial_size, double dt);

// Calculates energy of a configuration and stores it as the particle potential
void find_weights(h3plus& state);

// Random walk
void random_walk(h3plus& state);

// Branching scheme
void branch(h3plus& state);

// Random walking of matrix of position created in populate
void diffuse(h3plus& state, std::ostream& output);

void print_visualization_data(std::ostream& output, const h3plus& state);

template <typename T>
double random_double(T distribution);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
    std::ofstream output("out.dat", std::ostream::out);

    auto state = generate_initial(INITIAL_SIZE, TIMESTEP);
    //std::cout << state.v_ref << '\t' << state.particles.size() << '\n';

    diffuse(state, output);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Populate a distribution of particles for QMC
// Unlike Anderson, we are initilizing each psip randomly from a distribution.
// This might screw things up, because
h3plus generate_initial(size_t initial_size, double dt) {
    h3plus state(MAX_SIZE);
    state.dt = dt;
    state.v_ref = 0;
    state.energy = 0;
    state.global_id = 0;
    for (size_t i = 0; i < 3; i++){
        state.proton.pos[i][i] = 0.1;
    }

    // Random generation
    /*
    std::uniform_real_distribution<double> uniform(-1.0,1.0);
    for (size_t i = 0; i < initial_size; ++i){
        coord coords;
        for (auto& coord : coords) {
            coord = random_double(uniform);
        }
        state.particles.emplace_back(state.global_id++, std::move(coords));
    }
    */

    // Static generation
    for (size_t i = 0; i < initial_size; ++i) {
        coord coords = {RADIUS, RADIUS, RADIUS, -RADIUS, -RADIUS, -RADIUS};
        state.particles.emplace_back(state.global_id++, std::move(coords));
    }

    find_weights(state);
    return state;
}

// Calculates energy of a configuration and stores it into the final element of
// the MatrixPSIP
// Note: When calculating the potential, I am not sure whether we need to use
//       absolute value of distance or the distance, itself.
// Note: Inefficient. Can calculate energy on the fly with every generation of
//       psip. Think about it.
// Note: v_ref will be a dummy variable for now.
void find_weights(h3plus& state){
    // Note that this is specific to the Anderson paper
    // Finding the distance between electrons, then adding the distances
    // from the protons to the electrons.
    double potential_sum = 0.0, dist2;

    for (auto& particle : state.particles) {
        double dist1 = sqrt((particle.coords[0] - particle.coords[3]) *
                           (particle.coords[0] - particle.coords[3]) +
                           (particle.coords[1] - particle.coords[4]) *
                           (particle.coords[1] - particle.coords[4]) +
                           (particle.coords[2] - particle.coords[5]) *
                           (particle.coords[2] - particle.coords[5]));

        double potential = 1.0 / dist1;

        for (size_t i = 0; i < 3; i++){
            dist1 = sqrt((particle.coords[0] - state.proton.pos[i][0]) *
                         (particle.coords[0] - state.proton.pos[i][0]) +
                         (particle.coords[1] - state.proton.pos[i][1]) *
                         (particle.coords[1] - state.proton.pos[i][1]) +
                         (particle.coords[2] - state.proton.pos[i][2]) *
                         (particle.coords[2] - state.proton.pos[i][2]));

            dist2 = sqrt((particle.coords[3] - state.proton.pos[i][0]) *
                         (particle.coords[3] - state.proton.pos[i][0]) +
                         (particle.coords[4] - state.proton.pos[i][1]) *
                         (particle.coords[4] - state.proton.pos[i][1]) +
                         (particle.coords[5] - state.proton.pos[i][2]) *
                         (particle.coords[5] - state.proton.pos[i][2]));

            potential -= ((1/dist1) + (1/dist2));

        }
        /*
        for (const auto& coord : particle.coords) {
            
            potential -= 1.0 / std::abs(coord);
        }
        */
        particle.potential = potential;
        potential_sum += potential;
    }

    size_t num_particles = state.particles.size();
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // This needs type promotion to either double or int so it doesn't
    // underflow. Since it's going to be used as a double, might as well
    // do that directly.
    double particles_diff = static_cast<double>(num_particles) - INITIAL_SIZE;

    // defining the new reference potential
    state.energy = potential_sum / num_particles;
    state.v_ref = state.energy - particles_diff / (INITIAL_SIZE * state.dt);

    #pragma omp parallel for
    for (size_t i = 0; i < num_particles; ++i) {
        auto& particle = state.particles[i];
        double w = 1.0 - (particle.potential - state.v_ref) * state.dt;
        particle.m_n = std::min((int)(w + random_double(uniform)), 3);
    }
}

// 6D random walk
void random_walk(h3plus& state) {
    std::normal_distribution<double> gaussian(0.0, 1.0);
    size_t particles_size = state.particles.size();

    #pragma omp parallel for
    for (size_t i = 0; i < particles_size; ++i) {
        auto& particle = state.particles[i];
        for (auto& coord : particle.coords) {
            coord += sqrt(state.dt) * random_double(gaussian);
        }
    }
}

// Branching scheme
void branch(h3plus& state){
    find_weights(state);

    // I only cache this so I can fit the 80 character limit
    auto& particles = state.particles;

    // First, remove particles where M_n is zero, as per the Kosztin paper.
    // It's better to do it first to avoid unnecessary bookkeeping later
    auto remove = std::remove_if(std::begin(particles), std::end(particles),
                                 [](const particle& p) { return p.m_n == 0; });
    particles.erase(remove, std::end(particles));

    // Iterate over the current set of particles and add new ones as necessary.
    // The added particles will be skipped in the current iteration (which is
    // essentially the same behaviour as we had before)
    // We have to use direct indexes because of iterator invalidation.
    // Also note that current_size is cached outside of the loop, this is to
    // avoid unnecessary work
    size_t current_size = particles.size();
    for (size_t i = 0; i < current_size; ++i) {
        switch (particles[i].m_n) {
            case 2:
                particles.emplace_back(state.global_id++, particles[i]);
                break;
            case 3:
                particles.emplace_back(state.global_id++, particles[i]);
                particles.emplace_back(state.global_id++, particles[i]);
                break;
        }
    }

    // Just truncate if the vector would grow too large.
    // This should - theoratically - not happen
    if (particles.size() > MAX_SIZE) {
        particles.erase(std::begin(particles) + MAX_SIZE, std::end(particles));
    }
}

// Random walking of matrix of position created in populate
// Step 1: Move particles via 6D random walk
// Step 2: Destroy and create particles as need based on Anderson
// Step 3: check energy, end if needed.
void diffuse(h3plus& state, std::ostream& output){

    // For now, I am going to set a definite number of timesteps
    // This will be replaced by a while loop in the future.

    // double diff = 1.0;
    // while (diff > 0.01) {
    for (size_t t = 0; t < 1000; t++){
        double v_last = state.v_ref;

        random_walk(state);
        branch(state);
        // diff = sqrt((v_last - state.v_ref) * (v_last - state.v_ref));

        // Debug information for the current timestep
        std::cout << std::fixed
                  << state.v_ref << '\t'
                  << state.particles.size() << '\n';
        if (t % 10 == 0) {
            print_visualization_data(output, state);
        }
    }
}

void print_visualization_data(std::ostream& output, const h3plus& state) {
    for (const auto& particle : state.particles) {
        for (const auto& coord : particle.coords) {
            output << std::fixed << coord << "\t";
        }
        output << particle.m_n << "\t";
        output << particle.id << "\n";
    }
    output << '\n' << '\n';
}

static inline std::mt19937& random_engine() {
    static std::random_device rd;
    // Use the Mersenne twister, the default random engine is not guaranteed
    // to be high quality. Use engine(rd()) if you want to test different cases
    static thread_local std::mt19937 engine; // engine(rd());
    return engine;
}

template <typename T>
double random_double(T distribution) {
    return distribution(random_engine());
}
