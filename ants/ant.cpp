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
#include <algorithm>

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
    int stepnum, phenum, pernum, flag;
};

struct grid{
    array2d<bool, n, n> wall;
    coord prize;
};

// Functions for ant movement
// Chooses step
ant step(ant curr, grid landscape);

// Generates ants
std::vector<ant> gen_ants(std::vector<ant> ants, ant plate);

// Changes template ant
ant plate_toss(ant winner, coord spawn);

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
    landscape.prize.x = 1;   landscape.prize.y = n - 1;

    // defining spawn point
    coord spawn;
    spawn.x = n - 1;
    spawn.y = 0;

    // defining misc characters
    int final_time = 200;
    int pernum = 10;

    move(ants, landscape, spawn, pernum, final_time, output);

    output.close();

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Chooses step
ant step(ant curr, grid landscape){

    coord next_step[4];
    coord up, down, left, right, next;
    int pcount = 0;

    //std::cout << curr.pos.x << '\t' << curr.pos.y << '\n' << '\n';

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
        if (up.y < n && landscape.wall[up.x][up.y] == 0){
            next_step[pcount] = up;
            pcount++;
        }
    }

    // down case
    if (last != down) {
        if (down.y > 0 && landscape.wall[down.x][down.y] == 0){
            next_step[pcount] = down;
            pcount++;
        }
    }

    if (last != left) {
        if (left.x > 0 && landscape.wall[left.x][left.y] == 0){
            next_step[pcount] = left;
            pcount++;
        }
    }

    // right case
    if (last != right) {
        if (right.x < n && landscape.wall[right.x][right.y] == 0){
            next_step[pcount] = right;
            pcount++;
        }
    }

    static std::random_device rd;
    auto seed = rd();
    static std::mt19937 gen(seed);

    if (curr.flag == 0 && curr.phetus[curr.pos.x][curr.pos.y] == 0){
        std::uniform_int_distribution<int> ant_distribution(0,pcount - 1);

        next = next_step[ant_distribution(gen)];

        curr.antpath.push_back(curr.pos);
        curr.pos = next;
    }

    else{

        double prob = curr.pernum / curr.phenum, aprob[4], rn, psum = 0;
        int choice = -1, cthulu = 0;
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
            if (next_step[q].x == curr.phepath[cthulu].x &&
                next_step[q].y == curr.phepath[cthulu].y){
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
ant plate_toss(ant winner, coord spawn){

    ant plate;
    plate.pos.x = spawn.x;
    plate.pos.y = spawn.y;
    plate.phenum = winner.stepnum;
    plate.stepnum = 0;
    plate.phepath = winner.antpath;
    plate.phepath.push_back(winner.pos);

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
    int flag_tot = 0;

    // setting template for first generate
    ant plate;

    plate.pos.x = spawn.x;
    plate.pos.y = spawn.y;
    plate.stepnum = 0;
    plate.phenum = n*n;
    plate.phetus = {};
    plate.flag = 0;

    // to be defined later
    plate.pernum = pernum;

/*
    // define walls and prize at random spot
    grid landscape;
    landscape.wall = {};
    landscape.prize.x = n;   landscape.prize.y = n;
*/


    for (size_t i = 0; i < final_time; i++){
        //std::cout << i << '\n';
        int flag = 0;
        // step 1: generate ant
        ants = gen_ants(ants, plate);

        // step 2: Move ants
        for (size_t j = 0; j < ants.size(); j++){
            ants[j] = step(ants[j], landscape);
            if (ants[j].stepnum >= ants[j].phenum){
                killlist.push_back(j);
            }

            if (ants[j].pos.x == landscape.prize.x &&
                ants[j].pos.y == landscape.prize.y){
                plate = plate_toss(ants[j], spawn);
                plate.flag = 1;
                flag_tot = 1;
            }

        }

        std::reverse(std::begin(killlist), std::end(killlist));

        std::cout << "size: " << killlist.size() << '\n';
        for (size_t k = 0; k < killlist.size(); k++){
            ants.erase(ants.begin() + killlist[k]);
        }
        killlist.clear();

/*
        if (flag_tot == 0){
            for (size_t l = 0; l < ants[0].antpath.size(); l++){
                output << ants[0].antpath[l].x << '\t'
                          << ants[0].antpath[l].y << '\n';
            }
            output << '\n' << '\n';
        }
*/

        if (flag_tot == 1){
            for (size_t l = 0; l < ants[0].phepath.size(); l++){
                output << ants[0].phepath[l].x << '\t'
                          << ants[0].phepath[l].y << '\n';
            }
            output << '\n' << '\n';
        }
    }

    return ants;
}
