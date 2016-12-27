/*-------------worstsort.cpp--------------------------------------------------//
*
* Purpose: Implement worstsort for all the bad things to happen in 2016
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include "../visualization/cairo/cairo_vis.h"

// Function for worstsort
void worst_sort(frame &anim, std::vector<int> &array);

// Function for bubble sort
void bubble_sort(frame &anim, std::vector<int> &array);

// Function for Bogo sort
void bogo_sort(frame &anim, std::vector<int> &array);

// Function to determine whether an array is sorted
bool is_sorted(std::vector<int> &array);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Initializing cairo stuff
    std::vector<frame> layers(3);
    for (size_t i = 0; i < layers.size(); i++){
        layers[i].create_frame(400, 300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0], 0, 0, 0);

    // Defining color to use for lines
    color line_clr = {1, 1, 1, 1};


    // Random initialization of array
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int>
        dist(0,10000);

    int num_bars = 5;

    std::vector<int> array(num_bars);
    for (int i = 0; i < num_bars; i++){
        array[i] = dist(gen);
    }

    std::cout << std::is_sorted(std::begin(array), std::end(array)) << '\n';

    //bar_graph(layers[1], 10, 0, num_frames, array, layers[0].res_x,
    //          layers[0].res_y, line_clr);
    //bogo_sort(layers[1], array);
    bubble_sort(layers[1], array);

    draw_layers(layers);

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to determine whether array is sorted
bool is_sorted(std::vector<int> &array){

    int num1, num2;

    // Checking to see if adjacent values are ever in the wrong order.
    for (size_t i = 1; i < array.size(); i++){
        num1 = array[i-1];
        num2 = array[i];
        if (num2 < num1){
            return false;
        }
    }

    return true;

}

// Function to sort array based on bogo sort
void bogo_sort(frame &anim, std::vector<int> &array){

    color line_clr = {1, 1, 1, 1};

    // Shuffling everything until we have a sorted list
    while (!std::is_sorted(array.begin(), array.end())){
        std::next_permutation(array.begin(), array.end());
        bar_graph(anim, 0, anim.curr_frame, anim.curr_frame + 1, array, 
                  anim.res_x, anim.res_y, line_clr);
        anim.curr_frame++;
    }
}

// Function to sort array based on bubble sort
void bubble_sort(frame &anim, std::vector<int> &array){

    color line_clr = {1, 1, 1, 1};
/*
    for (size_t i = 0; i < array.size(); i++){
        for (size_t j = 0; j < array.size(); j++){
            if (array[i] < array[j]){
                std::swap(array[i], array[j]);
            }
            bar_graph(anim, 0, anim.curr_frame, anim.curr_frame + 1, array,
                      anim.res_x, anim.res_y, line_clr);
            anim.curr_frame++;
        }
    }
*/

    std::cout << std::is_sorted(array.begin(), array.end()) << '\n';
}
