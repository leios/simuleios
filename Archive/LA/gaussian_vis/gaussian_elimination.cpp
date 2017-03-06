/*-------------gaussian_elimination.cpp---------------------------------------//
*
* Purpose: Implement a simple version of gaussian elimination
* 
*   Notes: There is often a better alternative and there is no guarantee of
*          numerical stability
*          Visualization is transposed
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <utility>
#include "../../../visualization/cairo/cairo_vis.h"

// Function to swap individual rows of our matrix
void swap_rows(std::vector<double> &matrix, int row1, int row2, 
               int rows, int cols);

// Function to find maximum of a particular column
int max_col_index(std::vector<double> &matrix, int col, int k, 
                  int rows, int cols); 

// Function to print matrix
void print_matrix(std::vector<double> &matrix, int rows, int cols);

// Function for gaussian elimination
void gaussian_elimination(std::vector<double> &matrix, int rows, int cols);

// Function to swap elements (should use std::swap instead...)
void swap_elements(std::vector<double> &matrix, int ind1, int ind2);

// function to solve for our values with back-substitution
std::vector<double> backsubstitution(std::vector<double> matrix, 
                                     int rows, int cols);

// Function to visualize gaussian elimination algorithm of a matrix
void vis_matrix(frame &anim, std::vector<double> &matrix, int rows, int cols);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){

    // Creating matrix to work with 
    std::vector<double> matrix = {2, 3, 4, 6,
                                  1, 2, 3, 4,
                                  3, -4, 0, 10};

    print_matrix(matrix, 3, 4);

    gaussian_elimination(matrix, 3, 4);

    std::cout << '\n';

    print_matrix(matrix, 3, 4);

    std::cout << '\n';

    std::vector<double> solutions = backsubstitution(matrix, 3,4);

    for (int i = 0; i < solutions.size(); ++i){
        std::cout << solutions[i] << '\n';
    }

}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to swap individual rows of our matrix
void swap_rows(std::vector<double> &matrix, int row1, int row2, 
               int rows, int cols){

    for (int i = 0; i < cols; ++i){
        swap_elements(matrix, row1*cols + i, row2*cols + i);
    } 
}

// Function to find maximum of a particular column
int max_col_index(std::vector<double> &matrix, int col, int k,  
                  int rows, int cols){

    int max_index = k;
    for (int i = k; i < rows; ++i){
        if (abs(matrix[col + i*cols]) > abs(matrix[col + max_index*cols])){
            max_index = i;
        }
    }
    return max_index;
}

// Function to print matrix
void print_matrix(std::vector<double> &matrix, int rows, int cols){

    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            std::cout << matrix[i*cols + j] << '\t';
        }
        std::cout << '\n';
    }
}

// Function for gaussian elimination
void gaussian_elimination(std::vector<double> &matrix, int rows, int cols){

    // Initializing visualization stuff
    std::vector<frame> layers(3);
    for (size_t i = 0; i < layers.size(); i++){
        layers[i].create_frame(400, 300,30,"/tmp/image");
        layers[i].init();
        layers[i].curr_frame = 1;
    }
    create_bg(layers[0], 0, 0, 0);

    int max_index;
    double fraction;

    for (int k = 0; k < std::min(rows,cols); ++k){

        vis_matrix(layers[0], matrix, rows, cols);
        std::cout << '\n';

        max_index = max_col_index(matrix, k, k, rows, cols);
        if (matrix[max_index * cols + k] == 0){
            std::cout << "Matrix is singular!" << '\n';
            exit(1);
        }

        swap_rows(matrix, max_index, k, rows, cols);

        for (int i = k + 1; i < rows; ++i){
            fraction = matrix[i*cols + k] / matrix[k*cols +k];

            for (int j = k+1; j < cols; ++j){
                matrix[i*cols + j] -= matrix[k*cols + j] * fraction;
                //matrix[i*cols + k] = 0;
            }
            matrix[i*cols + k] = 0;
        }
    }

    draw_layers(layers);
}

// function to solve for our values with back-substitution
std::vector<double> backsubstitution(std::vector<double> matrix, 
                                     int rows, int cols){

    // Creating vector to work with
    std::vector<double> solutions(rows);

    solutions[rows-1] = matrix[(rows-1)*cols + cols-1] / 
                        matrix[(rows-1)*cols + cols-2];

    double sum = 0;
    for (int i = rows-2; i >= 0; --i){
        sum = 0;
        for (int j = rows-1; j > i; --j){
            sum += solutions[j] * matrix[i*cols + j];
        }
        solutions[i] = (matrix[i*cols + cols-1] - sum) / matrix[i*cols + i];
    }

    return solutions;
}

void swap_elements(std::vector<double> &matrix, int ind1, int ind2){

    double temp_val = matrix[ind1];
    matrix[ind1] = matrix[ind2];
    matrix[ind2] = temp_val;
}

// Function to visualize gaussian elimination algorithm of a matrix
void vis_matrix(frame &anim, std::vector<double> &matrix, int rows, int cols){

    vec pos;
    double box_size = 100;
    color square_clr;
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j){
            pos.x = anim.res_x * 0.5 + (-0.5 * rows + i) * box_size;
            pos.y = anim.res_y * 0.5 + (-0.5 * cols + j) * box_size;
            square_clr = {matrix[i*cols +j] / 10, 0,
                          1 - matrix[i*cols +j] / 10, 1};
            grow_square(anim, 0, anim.curr_frame, num_frames, pos, 
                        box_size, square_clr);
        }
    }

    anim.curr_frame++;
}
