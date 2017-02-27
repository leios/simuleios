/*-------------thomas.cpp-----------------------------------------------------//
*
* Purpose: Implement the Tri-Diagonal Matrix Algorithm (TDMA / Thomas Algorithm)
*          for a tridiagonal system of equations
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>

// Function for the Thomas algorithm
void thomas(std::vector<double> &a, std::vector<double> &b,
            std::vector<double> &c, std::vector<double> &soln, 
            std::vector<double> &d);

int main(){
    std::vector<double> a(3), b(3), c(3), soln(3), d(3);

    for (int i = 0; i < a.size(); ++i){
        a[i] = 0;
        c[i] = 0;
        b[i] = 1;
        d[i] = i;
    }

    thomas(a, b, c, soln, d);

    for (int i = 0; i < soln.size(); ++i){
        std::cout << soln[i] << '\n';
    }
}

// Function for the Thomas algorithm
void thomas(std::vector<double> &a, std::vector<double> &b,
            std::vector<double> &c, std::vector<double> &soln, 
            std::vector<double> &d){

    c[0] /= b[0];
    d[0] /= b[0];

    double id;
    for (int i = 1; i < soln.size(); ++i){
        id = 1 / (b[i] - c[i-1] * a[i]);
        c[i] *= id;
        d[i] = d[i] - a[i] * d[i-1] * id;
    }

    soln[soln.size()-1] = d[soln.size()-1];

    for (int i = a.size()-2; i > -1; --i){
        soln[i] = d[i] - c[i] * soln[i+1];
    }
}
