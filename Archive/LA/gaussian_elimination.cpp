/*-------------gaussian_elimination.cpp---------------------------------------//
*
* Purpose: Implement a simple version of gaussian elimination
* 
*   Notes: There is often a better alternative and there is no guarantee of
*          numerical stability
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <utility>

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

// function to solve for our values with back-substitution
std::vector<double> backsubstitution(std::vector<double> matrix, 
                                     int rows, int cols);

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
               int cols, int rows){

    for (int i = 0; i < cols; ++i){
        std::swap(matrix[row1*cols + i], matrix[row2*cols + i]);
    } 
}

// Function to find maximum of a particular column
int max_col_index(std::vector<double> &matrix, int col, int k,  
                  int rows, int cols){

    int max_index = k;
    for (int i = k; i < rows; ++i){
        if (matrix[col + i*cols] > matrix[max_index + i*cols]){
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

    int max_index;
    double fraction;

    for (int k = 0; k < std::min(rows,cols); ++k){

        max_index = max_col_index(matrix, k, k, rows, cols);
        if (matrix[max_index * cols, k] == 0){
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
