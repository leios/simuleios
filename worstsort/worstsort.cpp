/*-------------worstsort.cpp--------------------------------------------------//
*
* Purpose: Implement worstsort for all the bad things to happen in 2016
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "../visualization/cairo/cairo_vis.h"

// Necessary functions for worstsort
bool lt(std::vector<int> &A, std::vector<int> &B);
std::vector<std::vector<int>> perm(std::vector<int> &L);

// Function for worstsort
void worst_sort(frame &anim, int depth, std::vector<int> &array);

// Function for bubble sort
void bubble_sort(frame &anim, std::vector<int> &array);
void bubble_sort(frame &anim, std::vector<std::vector<int>> &array);

// Function for Bogo sort
void bogo_sort(frame &anim, std::vector<int> &array);

// Function to determine whether an array is sorted
bool is_sorted(std::vector<int> &array);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Testing the lt function
    std::vector<int> A = {1,2,3};
    std::vector<int> B = {3,2,1};
    std::cout << lt(A, B) << '\t' << lt(B,A) << '\n';

    // Testing the permutation function
    std::vector<std::vector<int>> P = perm(A);

    for (size_t i = 0; i < P.size(); i++){
        for (size_t j = 0; j < P[0].size(); j++){
            std::cout << P[i][j] << '\t';
        }
        std::cout << '\n';
    }

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

    int num_bars = 25;

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
    color highlight_clr = {1, 0, 0, 1};
    for (size_t i = 1; i < array.size(); i++){
        for (size_t j = 1; j < array.size() - i + 1; j++){
            if (array[j] < array[j-1]){
                std::swap(array[j], array[j-1]);
            }
            bar_graph(anim, 0, anim.curr_frame, anim.curr_frame + 1, array,
                      anim.res_x, anim.res_y, line_clr);
            highlight_bar(anim, anim.curr_frame, anim.curr_frame+1, array, 
                          anim.res_x, anim.res_y, highlight_clr, j);
            highlight_bar(anim, anim.curr_frame, anim.curr_frame+1, array, 
                          anim.res_x, anim.res_y, highlight_clr, j-1);
            anim.curr_frame++;
        }
    }

    std::cout << std::is_sorted(array.begin(), array.end()) << '\n';
}

// Auxiliary functions for worstsort
bool lt(std::vector<int> &A, std::vector<int> &B){
    // Check that A and B are the same size
    if (A.size() != B.size()){
        std::cout << "The array sizes are not equal. You screwed up!" << '\n';
        assert(A.size() == B.size());
    }

    // Checking each element, until a pair is different. Then we return the 
    // value for A[i] < B[i]. If we get to the end, all elements are equal...
    for (size_t i = 0; i < A.size(); i++){
        if (A[i] < B[i]){
            return true;
        }
        if (A[i] > B[i]){
            return false;
        }
    }

    return false;
}

// Function to find list of lists for all permutations 
std::vector<std::vector<int>> perm(std::vector<int> &L){
    std::vector<std::vector<int>> P, P0;
    std::vector<int> L1, L_check;
    if (L.size() <= 1){
        P.push_back(L);
        return P;
    }
    else{
        for (size_t i = 0; i < L.size(); i++){
            L1 = L;
            L1.erase(L1.begin()+i);
            P0 = perm(L1);
            for (size_t j = 0; j < P0.size(); j++){
                P0[j].insert(P0[j].begin(), L[i]);
                P.push_back(P0[j]);
                
            }
        }
    }

    return P;
}

// Function for worstsort
void worst_sort(frame &anim, int depth, std::vector<int> &array){

    std::vector<std::vector<int>> P;
    if (depth == 0){
        bubble_sort(anim, array);
    }
    else{
        P = perm(array);
        std::vector<int> index(P.size());
        for (size_t i = 0; i < index.size(); i++){
            index[i] = i;
        }
        worst_sort(anim, depth - 1, index);
        array = P[0];
    }
}
