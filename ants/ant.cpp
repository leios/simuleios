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

/*----------------------------------------------------------------------------//
* STRUCTS
*-----------------------------------------------------------------------------*/

template <typename T, size_t rows, size_t cols>
using array2d = std::array<std::array<T, cols>, rows>;

// grid size
const int n = 5;

// structs for ant movement
struct coord{
    int x, y;
};

inline bool operator==(const coord& lhs, const coord& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline bool operator!=(const coord& lhs, const coord& rhs) {
    return !(lhs == rhs);
}

struct ant{
    coord pos;
    std::vector<coord> phepath, antpath;
    int step, phenum, pernum;
};

struct grid{
    array2d<bool, n, n> wall;
    coord prize;
};

// Functions for ant movement
// Chooses step
ant step(ant curr, grid landscape, coord spawn_point, int gen_flag);

// Generates ants
std::vector<ant> gen_ants(coord spawn_point);

// Moves simulation every timestep
std::vector<ant> move(std::vector <ant> ants, grid landscape);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Chooses step
ant step(ant curr, grid landscape, coord spawn_point, int gen_flag){

    coord next_step[3];
    coord up, down, left, right, next;
    int pcount = 0;

    up.x = curr.pos.x;          up.y = curr.pos.y + 1;
    down.x = curr.pos.x;        down.y = curr.pos.y - 1;
    left.x = curr.pos.x - 1;    left.y = curr.pos.y;
    right.x = curr.pos.x + 1;   right.y = curr.pos.y;

    coord last = curr.antpath.back();
    // determine possible movement
    // up case
    if (last != up) {
        next_step[pcount] = up;
        pcount++;
    }

    // down case
    if (last != down) {
        next_step[pcount] = down;
        pcount++;
    }

    if (last != left) {
        next_step[pcount] = left;
        pcount++;
    }

    // right case
    if (last != right) {
        next_step[pcount] = right;
        pcount++;
    }

    static std::random_device rd;
    auto seed = rd();
    static std::mt19937 gen(seed);

    if (gen_flag == 0){
        std::uniform_int_distribution<int> ant_distribution(0,3);

        next = next_step[ant_distribution(gen)];

        curr.antpath.push_back(curr.pos);
        curr.pos = next;
    }

// WORKING ON THIS TOMORROW
/*
    else{

        double prob = curr.pernum / curr.phenum;
        int choice;
        std::uniform_real_distribution<double> ant_distribution(0,1);

        next = next_step[choice];

        curr.antpath.push_back(curr.pos);
        curr.pos = next;

    }

*/
    return curr;


}

