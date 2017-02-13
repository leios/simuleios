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
std::vector<Chromosome> crossover(std::vector<Chromosome> &parents, 
                                  double cross_rate);

// Function to return the location of a provided city on the chromosome
int ch_loc(Chromosome &parent, std::string city, int ignore);

// Function to find the elite of the previous generation
std::vector<Chromosome> find_elite(std::vector<Chromosome> &population,
                                  int elite_num);

// Function for repopulation
std::vector<Chromosome> repopulate(std::vector<Chromosome> &offspring,
                                   std::vector<Chromosome> &elite);

// Function to find the fitness
void find_fitness(std::vector<Chromosome> &population);

// Function to output the current city path
void output_path(Chromosome &ch);

// Function to implement mutation
void mutate(std::vector<Chromosome> &population, double mut_rate);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    int generations = 100;
    int pop_size = 100;

    // Now we just need to put it all together!
    std::vector<City> cities = init_cities();

    std::vector<Chromosome> population = init(cities, pop_size);
    std::vector<Chromosome> offspring, parents, elite;
    for (int i = 0; i < generations; ++i){

        std::cout << i << '\n';
        find_fitness(population);
        parents = selection(population, 10, 100);
        offspring = crossover(parents, 0.5);
        elite = find_elite(population, 5);
        population = repopulate(offspring, elite);
        mutate(population, 0.10);
    }

    for (int i = 0; i < population.size(); ++i){
        for (int j = 0; j < population[i].path.size(); ++j){
            std::cout << population[i].path[j]->name ;
        }
        std::cout << '\n';
    }
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
        City("E", {0.25, 0.25}),
        City("F", {0.125, 0.625}),
        City("G", {0.25, 0.5})
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

    for (int i = 0; i < t_size; i++){
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
std::vector<Chromosome> crossover(std::vector<Chromosome> &parents, 
                                  double cross_rate){

    // We need to initialize the children first
    std::vector<Chromosome> offspring = parents;

    // The crossover step here will produce 2 children from 2 parents, 
    // And we will be using std::swap for this.

    // This will basically be a single-point crossover, so we need to randomly
    // Decide at which point we will be performing the crossover

    static std::random_device rd;
    static std::mt19937 gen(rd());

    static std::uniform_int_distribution<int>
        cross_pt_dist(0,parents[0].path.size() - 1);

    Chromosome child1;
    Chromosome child2;

    int cross_pt = 0;
    std::string cross_city1, cross_city2;

/*
    std::cout << "outputting all parents" << '\n';
    for (int i = 0; i < parents.size(); ++i){
        output_path(parents[i]);
    }
*/

    for (int i = 0; i < offspring.size(); i += 2){

        if (std::rand() % 100 * 0.01 < cross_rate){

            //in here we need to do the crossover.

            // First, we need to find the location of the initial swap
            cross_pt = cross_pt_dist(gen);
            //std::cout << "cp is: " << cross_pt << '\n';
            //std::cout << "i is: " << i << '\n';
            cross_city1 = offspring[i].path[cross_pt]->name;
            cross_city2 = offspring[i+1].path[cross_pt]->name;
            //std::cout << "match this: "<< cross_city1 << '\n';
            //std::cout << "with: "<< cross_city2 << '\n';
            std::swap(offspring[i].path[cross_pt], 
                      offspring[i+1].path[cross_pt]);
    
            while(cross_city1 != cross_city2 || cross_pt != -1){
                //std::cout << offspring[i].path[cross_pt]->name << '\n';
                //std::cout << cross_pt << '\t' << cross_city2 << '\n';
                //output_path(offspring[i]);
                //output_path(offspring[i+1]);
                cross_city2 = offspring[i+1].path[cross_pt]->name;
                std::swap(offspring[i].path[cross_pt], 
                          offspring[i+1].path[cross_pt]);
                cross_pt = ch_loc(offspring[i], cross_city2, cross_pt);
            }
    
            //std::cout << "while done " << i << '\n';
        }
    }

/*
    std::cout << "outputting all offspring" << '\n';
    for (int i = 0; i < offspring.size(); ++i){
        output_path(offspring[i]);
    }
*/

    return offspring;
}

// Function to return the location of a provided city on the chromosome
int ch_loc(Chromosome &parent, std::string city, int ignore){

    int loc = -1;

    for (int i = 0; i < parent.path.size(); ++i){
        if (i != ignore){
            if (parent.path[i]->name == city){
                return i;
            }
        }
    }

    return loc;
}

// Function to find the elite of the previous generation
std::vector<Chromosome> find_elite(std::vector<Chromosome> &population,
                                  int elite_num){

    // First, we need to sort the array to find the best of the best!
    std::vector<Chromosome> sorted_pop(population.size());

    sorted_pop = population;
    std::sort(sorted_pop.begin(), sorted_pop.end(),
              [](const Chromosome &i, const Chromosome &j){
                  return i.fitness < j.fitness;});

    std::vector<Chromosome> elite(elite_num);
    for (int i = 0; i < elite_num; ++i){
        elite[i] = sorted_pop[i];
        std::cout << sorted_pop[i].fitness << '\n';
    }

    std::cout << '\n';

    return elite;
    
}

// Function for repopulation
std::vector<Chromosome> repopulate(std::vector<Chromosome> &offspring,
                                   std::vector<Chromosome> &elite){

    std::vector<Chromosome> new_population(offspring.size());

    for (size_t i = 0; i < elite.size(); ++i){
        new_population[i] = elite[i];
    }

    for (size_t i = elite.size(); i < offspring.size(); ++i){
        new_population[i] = offspring[i];
    }

    return new_population;
}

// function to find the fitness of each individual in the population
void find_fitness(std::vector<Chromosome> &population){

    double dist = 0;
    for (auto &ind : population){
        dist = 0;
        for (int i = 1; i < ind.path.size(); ++i){
            dist += distance(ind.path[i-1]->loc, ind.path[i]->loc);
        }

        ind.fitness = dist;
    }
}

// Function to output the current city path
void output_path(Chromosome &ch){

    for (int i = 0; i < ch.path.size(); ++i){
        std::cout << ch.path[i]->name;
    }

    std::cout << '\n';
}

// Function to implement mutation
void mutate(std::vector<Chromosome> &population, double mut_rate){

    static std::random_device rd;
    static std::mt19937 gen(rd());

    static std::uniform_int_distribution<int>
        cross_pt_dist(0,population[0].path.size() - 1);
    
    int cross_pt1;
    int cross_pt2;

    for (auto &ind : population){
        output_path(ind);
        if (std::rand() % 100 * 0.01 < mut_rate){
            cross_pt1 = cross_pt_dist(gen);
            cross_pt2 = cross_pt_dist(gen);
            std::swap(ind.path[cross_pt1], 
                      ind.path[cross_pt2]);
        }
        
    }
}
