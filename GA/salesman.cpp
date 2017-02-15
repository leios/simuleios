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

// Visualization functions
// Function to take individual along every town in their path
void move(frame &anim, Chromosome &individual, int start_frame, int end_frame,
          color &human_clr);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Initialize all visualization components
    int fps = 30;
    double res_x = 1000;
    double res_y = 1000;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 1};
    color line_clr = {1, 0, 0, 1};
    color white = {1, 1, 1, 1};
    color cir_clr = {0, 0, 0, 1};

    std::vector<frame> layers = init_layers(1, res, fps, bg_clr);

    // now we need to draw the exterior region of BetaBrook

    // Let's start by defining an exterior region
    std::vector<vec> bb_region(13);
    bb_region[0] = {0.05,0.05};
    bb_region[1] = {0.25,0.10};
    bb_region[2] = {0.75,0.03};
    bb_region[3] = {0.95,0.05};
    bb_region[4] = {0.95,0.25};
    bb_region[5] = {0.85,0.65};
    bb_region[6] = {0.95,0.95};
    bb_region[7] = {0.55,0.9};
    bb_region[8] = {0.45,0.95};
    bb_region[9] = {0.05,0.95};
    bb_region[10] = {0.05,0.95};
    bb_region[11] = {0.10,0.5};
    bb_region[12] = {0.05,0.05};

    draw_array(layers[0], 0.1, bb_region, res_x, res_y, line_clr);

    int generations = 100;
    int pop_size = 100;

    // Now we just need to put it all together!
    std::vector<City> cities = init_cities();

    // Now we need to draw the cities

    vec cir_loc = {0,0};
    std::string text;
    for (int i = 0; i < cities.size(); i++){
        cir_clr = {(double)i/cities.size(),0,
                   (double)(cities.size()-i)/cities.size(),1};

        cir_loc.x = cities[i].loc.x * res_x;
        cir_loc.y = (1-cities[i].loc.y) * res_y;
        grow_circle(layers[0], 0.1, cir_loc, 20, cir_clr);

        // writing single letter name on all subsequent frames
        for (int j = layers[0].curr_frame; j < num_frames; ++j){

            text = cities[i].name;
            cairo_set_font_size(layers[0].frame_ctx[j], 30);
            cairo_set_source_rgba(layers[0].frame_ctx[j],
                                  white.r, white.g, white.b, white.a);

            // Determining where to move to
            cairo_text_extents_t textbox;
            cairo_text_extents(layers[0].frame_ctx[j], 
                               text.c_str(), &textbox);
            cairo_move_to(layers[0].frame_ctx[j], 
                          cir_loc.x - textbox.width / 2.0, 
                          cir_loc.y + textbox.height * 0.5);
            cairo_show_text(layers[0].frame_ctx[j], 
                            text.c_str());
        
            cairo_stroke(layers[0].frame_ctx[layers[0].curr_frame]);


        }

    }

    layers[0].curr_frame += 10;

    // Now let's draw a human at the right position
    vec A_pos;
    A_pos.x = cities[0].loc.x * res_x;
    A_pos.y = (1-cities[0].loc.y) * res_y - 40;
    int start_frame = layers[0].curr_frame;
    animate_human(layers[0], A_pos, 40, white, start_frame,
                  start_frame + 50, 1);

    layers[0].curr_frame = start_frame+50;

    // now we need to create a Chromosome that goes ABCDEFG
    Chromosome test;

    for (int i = 0; i < cities.size(); ++i){
        test.path.push_back(&cities[i]);
    }

    // testing move
    move(layers[0], test, layers[0].curr_frame, num_frames, white);

/*
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
*/

    // We need to test them all here

    draw_layers(layers);
    for (int i = 0; i < layers.size(); ++i){
        layers[i].destroy_all();
    }

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to create initial list of cities, positions predefined
std::vector<City> init_cities(){

    std::vector<City> cities = {

        City("A", {0.5, 0.75}),
        City("B", {0.25, 0.75}),
        City("C", {0.5, 0.5}),
        City("D", {0.75, 0.75}),
        City("E", {0.25, 0.25}),
        City("F", {0.75, 0.25}),
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

        std::shuffle(genes.begin()+1, genes.end(), gen);
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
        cross_pt_dist(1,parents[0].path.size() - 1);

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
        cross_pt_dist(1,population[0].path.size() - 1);
    
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

// Visualization functions
// Function to take individual along every town in their path
void move(frame &anim, Chromosome &individual, int start_frame, int end_frame,
          color &human_clr){

    // The trick here is figuring out how to interpolate

    int draw_frames = end_frame - start_frame;

    int res = floor(draw_frames / individual.path.size());

    std::cout << "res is: " << res << '\n';

    // this is a counter for which leg of the race we are on
    vec curr_pos = {0,0};

    for (int i = 0; i < individual.path.size(); ++i){
        for (int j = 0; j < res; ++j){
            if (i+1 < individual.path.size()){
                curr_pos.x = individual.path[i]->loc.x + 
                             ((double)j/(res-1))*(individual.path[i+1]->loc.x -
                                              individual.path[i]->loc.x);
                curr_pos.y = 1-(individual.path[i]->loc.y + 
                             ((double)j/(res-1))*(individual.path[i+1]->loc.y -
                                              individual.path[i]->loc.y));
                curr_pos.x *= anim.res_x;
                curr_pos.y *= anim.res_y;
                curr_pos.y -= 40;
            }
            else{
                curr_pos.x = individual.path[i]->loc.x + 
                             ((double)j/(res-1))*(individual.path[0]->loc.x -
                                              individual.path[i]->loc.x);
                curr_pos.y = 1-(individual.path[i]->loc.y + 
                             ((double)j/(res-1))*(individual.path[0]->loc.y -
                                              individual.path[i]->loc.y));
                curr_pos.x *= anim.res_x;
                curr_pos.y *= anim.res_y;
                curr_pos.y -= 40;
            }
            draw_human(anim, curr_pos, 40, human_clr);
            //animate_human(anim, curr_pos, 40, human_clr, anim.curr_frame,
                          //anim.curr_frame+1, 0);
            anim.curr_frame++;
        }
    }

}
