/*------------ant.cpp---------------------------------------------------------//
*
*             ants -- We have them
*
* Purpose: Figure out ant pathing to food. I want food.
*
*-----------------------------------------------------------------------------*/

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

struct ant{
    coord pos;
    std::vector<coord> phepath, antpath;
    int step, phenum, pernum;
};

struct grid{
    bool wall[n][n];
    coord prize;
};

// Functions for ant movement
// Chooses step
ant step(ant curr, grid landscape, coord spawn_point, int gen_flag);

// Generates ants
std::vector <ant> gen_ants(coord spawn_point);

// Moves simulation every timestep
std::vector <ant> move(std::vector <ant> ants, grid landscape);

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

    // determine possible movement
    for (int i = 0; i < 4; i++){
        switch(i){
        // up case
        case 0:
            if (up.x == curr.antpath.back().x &&
                up.y == curr.antpath.back().y){
            }
            else{
                next_step[pcount] = up;
                pcount++;
            }
            break;

        // down case
        case 1:
            if (down.x == curr.antpath.back().x &&
                down.y == curr.antpath.back().y){
            }
            else{
                next_step[pcount] = down;
                pcount++;
            }
            break;

        // left case
        case 2:
            if (left.x == curr.antpath.back().x &&
                left.y == curr.antpath.back().y){
            }
            else{
                next_step[pcount] = left;
                pcount++;
            }
            break;

        // right case
        case 3:
            if (right.x == curr.antpath.back().x &&
                right.y == curr.antpath.back().y){
            }
            else{
                next_step[pcount] = right;
                pcount++;
            }
            break;

        }

    }

    static std::random_device rd;
    int seed = rd();
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

