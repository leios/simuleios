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
        if (landscape.wall[up.x][up.y] == 0){
            next_step[pcount] = up;
            pcount++;
        }
    }

    // down case
    if (last != down) {
        if (landscape.wall[down.x][down.y] == 0){
            next_step[pcount] = up;
            pcount++;
        }
    }

    if (last != left) {
        if (landscape.wall[left.x][left.y] == 0){
            next_step[pcount] = up;
            pcount++;
        }
    }

    // right case
    if (last != right) {
        if (landscape.wall[right.x][right.y] == 0){
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

        double prob = curr.pernum / curr.phenum, aprob[pcount], rn, psum = 0;
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

    return curr;


}

