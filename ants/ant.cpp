/*------------ant.cpp---------------------------------------------------------//
*
*             ants -- We have them
*
* Purpose: Figure out ant pathing to food. I want food.
*
*-----------------------------------------------------------------------------*/

#include <array>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

/*----------------------------------------------------------------------------//
* STRUCTS
*-----------------------------------------------------------------------------*/

// grid size
const int n = 5;

// structs for ant movement
struct coord{
    int x, y;
};

// Template for 2d boolean arrays 
template <typename T, size_t rows, size_t cols>
using array2d = std::array<std::array<T, cols>, rows>;

// defines operators for definitions above
inline bool operator==(const coord& lhs, const coord& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline bool operator!=(const coord& lhs, const coord& rhs) {
    return !(lhs == rhs);
}

struct ant{
    array2d<bool, n, n> phetus;
    coord pos;
    std::vector<coord> phepath, antpath;
    int stepnum, phenum, pernum;
};

struct grid{
    array2d<bool, n, n> wall;
    coord prize;
};

// Functions for ant movement
// Chooses step
ant step(ant curr, grid landscape, int gen_flag);

// Generates ants
std::vector<ant> gen_ants(std::vector<ant> ants, ant plate);

// Changes template ant
ant plate_toss(ant winner);

// Moves simulation every timestep
std::vector<ant> move(std::vector <ant> ants, grid landscape, coord spawn,
                      int pernum, int final_time, std::ofstream &output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // defining output file
    std::ofstream output("out.dat", std::ofstream::out);

    // defining initial ant vector and grid
    std::vector<ant> ants;
    grid landscape;
    landscape.wall = {};
    landscape.prize.x = n;   landscape.prize.y = n;

    // defining spawn point
    coord spawn;
    spawn.x = n; 
    spawn.y = 0;

    // defining misc characters
    int final_time = 10;
    int pernum = 10;

    move(ants, landscape, spawn, pernum, final_time, output);

    //output.close();

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Chooses step
ant step(ant curr, grid landscape, int gen_flag){

    coord next_step[4];
    coord up, down, left, right, next;
    int pcount = 0;

    std::cout << curr.pos.x << '\t' << curr.pos.y << '\n' << '\n';

    up.x = curr.pos.x;          up.y = curr.pos.y + 1;
    down.x = curr.pos.x;        down.y = curr.pos.y - 1;
    left.x = curr.pos.x - 1;    left.y = curr.pos.y;
    right.x = curr.pos.x + 1;   right.y = curr.pos.y;

    //std::cout << up.x << '\t' << up.y << '\n';

    coord last;
    if (curr.stepnum == 0){
        last.x = -1;
        last.y = -1;
    }
    else{
        last = curr.antpath.back();
    }

    // determine possible movement
    // up case
    if (last != up) {
        if (landscape.wall[up.x][up.y] == 0 && up.y <= n){
            next_step[pcount] = up;
            pcount++;
        }
    }

    // down case
    if (last != down) {
        if (landscape.wall[down.x][down.y] == 0 && down.y >= 0){
            next_step[pcount] = up;
            pcount++;
        }
    }

    if (last != left) {
        if (landscape.wall[left.x][left.y] == 0 && left.x >= 0){
            next_step[pcount] = up;
            pcount++;
        }
    }

    // right case
    if (last != right) {
        if (landscape.wall[right.x][right.y] == 0 && right.x <= n){
            next_step[pcount] = up;
            pcount++;
        }
    }

    static std::random_device rd;
    auto seed = rd();
    static std::mt19937 gen(seed);

    if (gen_flag == 0 && curr.phetus[curr.pos.x][curr.pos.y] == 0){
        std::uniform_int_distribution<int> ant_distribution(0,pcount);

        next = next_step[ant_distribution(gen)];

        curr.antpath.push_back(curr.pos);
        curr.pos = next;
    }

    else{

        double prob = curr.pernum / curr.phenum, aprob[4], rn, psum = 0;
        int choice = -1, cthulu;
        std::uniform_real_distribution<double> ant_distribution(0,1);

        // search through phepath to find ant's curr location
        for (size_t q = 0; q < curr.phepath.size(); q++){
            if (curr.pos.x == curr.phepath[q].x &&
                curr.pos.y == curr.phepath[q].y){
                cthulu = q;
            }
        }

        std::uniform_real_distribution<double> ant_ddist(0,1);

        rn = ant_ddist(gen);

        for (size_t q = 0; q < pcount; q++){
            if (next_step[q].x == curr.phepath[cthulu +1].x &&
                next_step[q].y == curr.phepath[cthulu +1].y){
                aprob[q] = prob;
            }
            else{
                aprob[q] = (1 - prob) / (pcount - 1);
            }

            psum += aprob[q];

            if (rn < psum && choice < 0){
                choice = q;
            }

        }

        next = next_step[choice];

        curr.antpath.push_back(curr.pos);
        curr.pos = next;

    }
    curr.stepnum++;

    return curr;
}

// Generates ants
std::vector<ant> gen_ants(std::vector<ant> ants, ant plate){
    ant curr;
    curr = plate;

    ants.push_back(curr);

    return ants;
}

// Changes template ant
ant plate_toss(ant winner){

    ant plate = winner;
    plate.phenum = winner.stepnum;
    plate.stepnum = 0;

    // generate a new phetus
    plate.phetus = {};

    for (size_t i = 0; i < winner.antpath.size(); i++){
        plate.phetus[winner.antpath[i].x][winner.antpath[i].y] = 1;
    }

    plate.antpath = {};

    return plate;
}

// Moves simulation every timestep
// step 1: create ants
// step 2: move ants
// step 3: profit?
//         redefine all paths = to shortest path
std::vector<ant> move(std::vector <ant> ants, grid landscape, coord spawn,
                      int pernum, int final_time, std::ofstream &output){

    std::vector<int> killlist;

    // setting template for first generate
    ant plate;

    plate.pos.x = spawn.x;
    plate.pos.y = spawn.y;
    plate.stepnum = 0;
    plate.phenum = n*n;
    plate.phetus = {};

    // to be defined later
    plate.pernum = pernum;

/*
    // define walls and prize at random spot
    grid landscape;
    landscape.wall = {};
    landscape.prize.x = n;   landscape.prize.y = n;
*/
    
    
    for (size_t i = 0; i < final_time; i++){
        std::cout << i << '\n';
        int flag = 0;
        // step 1: generate ant
        ants = gen_ants(ants, plate);

        // step 2: Move ants
        for (size_t j = 0; j < ants.size(); j++){
            ants[j] = step(ants[j], landscape, flag);
            if (ants[j].stepnum > ants[j].phenum){
                killlist.push_back(j);
            }

            if (ants[j].pos.x == landscape.prize.x &&
                ants[j].pos.y == landscape.prize.y){
                plate = plate_toss(ants[j]);
            }

        }

        for (size_t k = 0; k < killlist.size(); k++){
            ants.erase(ants.begin() + killlist[k]);
        }

        for (size_t l = 0; l < ants[0].phepath.size(); l++){
            output << ants[0].phepath[l].x << '\t' 
                   << ants[0].phepath[l].y << '\n';
        }
        output << '\n' << '\n';
    }

    return ants;
}
