/*-------------salesman.cpp---------------------------------------------------//
*
* Purpose: To solve the travelling saleman problem with genetic algorithms
*
*   Notes: Not settled that c++ is the best language for this
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "../visualization/cairo/cairo_vis.h"

struct City{
    std::string name;
    vec loc;

    City(const std::string &n, vec l) : name(n), loc(l) { }
};

struct Chromosome{
    double fitness;
    std::vector<City*> path;
};

// Function to create initial list of cities, positions predefined
std::vector<City> init_cities();

// Function to initialize the population as random "walks" through cities
std::vector<Chromosome> init(std::vector<City> &cities, int size);

// Function to perform tournament selection
Chromosome tournament(std::vector<Chromosome> &population, int t_size);

// Function to perform tournament selection of our population
std::vector<Chromosome> selection(std::vector<Chromosome> &population,
                                  int t_size, int num_parents);

// Function for crossover
std::vector<Chromosome> crossover(std::vector<Chromosome> &parents, int size,
                                  std::vector<Chromosome> &elite);

// Function for mutation
std::vector<Chromosome> mutate(std::vector<Chromosome> &offspring);

// Function for repopulation
std::vector<Chromosome> repopulate(std::vector<Chromosome> &offspring,
                                   std::vector<Chromosome> &elite);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to create initial list of cities, positions predefined
std::vector<City> init_cities(){

    std::vector<City> cities = {

        City("A", {0.75, 0.25}),
        City("B", {0.25, 0.75}),
        City("C", {0.5, 0.5}),
        City("D", {0.75, 0.75}),
        City("E", {0.25, 0.25})
    };

    return cities;
}

// Function to initialize the population as random "walks" through cities
std::vector<Chromosome> init(std::vector<City> &cities, int size){

    // Initializing random device
    std::random_device rd;
    std::mt19937 gen(rd());

    // initializing population set
    std::vector<Chromosome> population(size);

    // initializing list of cities to work with
    std::vector<City*> genes;

    for (int i = 0; i < cities.size(); ++i){
        genes.push_back(&cities[i]);
    }

    // We need to start and end at the same city while also visiting all others
    // We will do this by shuffling our path / genes
    for (int i = 0; i < size; i++){

        std::shuffle(genes.begin(), genes.end(), gen);
        population[i].path = genes;
    }

    return population;
}

// Function to perform tournament selection
Chromosome tournament(std::vector<Chromosome> &population, int t_size){

    // Creating random number generator for selection
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> 
        choice(0, population.size() -1);

    Chromosome best = population[choice(gen)];
    Chromosome candidate;

    for (int i = 2; i < population.size(); i++){
        candidate = population[choice(gen)];
        if (candidate.fitness < best.fitness){
            best = candidate;
        }
    }

    return best;
}

// Function to perform tournament selection of our population
std::vector<Chromosome> selection(std::vector<Chromosome> &population,
                                  int t_size, int num_parents){
    std::vector<Chromosome> parents(num_parents);

    // selecting each individual parent
    for (int i = 0; i < num_parents; i++){
        parents[i] = tournament(population, t_size);
    }

    return parents;
}

// Function for crossover
std::vector<Chromosome> crossover(std::vector<Chromosome> &parents, int size,
                                  std::vector<Chromosome> &elite){

    // We need to initialize the children first
    std::vector<Chromosome> offspring(size);

    // The crossover step here will produce 2 children from 2 parents, 
    // And we will be using std::swap for this.

    // This will basically be a single-point crossover, so we need to randomly
    // Decide at which point we will be performing the crossover

    static std::random_device rd;
    static std::mt19937 gen(rd());

    static std::uniform_int_distribution<int>
        cross_pt(0,size-1);

    Chromosome child1;
    Chromosome child2;

    for (int i = 0; i < size; i += 2){

        //in here we need to do the crossover.

        offspring[i] = child1;
        offspring[i+1] = child2;
    }

    return offspring;
}

