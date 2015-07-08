/*-------------fxintegrate----------------------------------------------------//
*
*              Function integration
*
* Purpose: To integrate a random polynomial given in a string
*
*   Notes: Good luck! Have Fun!
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include<string>
#include<vector>
#include<stdlib.h>
#include<math.h>
#include<stdio.h>

using namespace std;

double integrate_poly(string polynomial, int resolution, double bound_a,
                      double bound_b);

int main(){

    double final = integrate_poly("3x^3^-2x^2^+43x^10^-1", 100, 0, 1);

    cout << final << endl;

}

double integrate_poly(string polynomial, int resolution, double bound_a,
                      double bound_b){

    vector<double> poly_double;
    double height = 0, integral, num, power, x, y, offset;
    int count = 0;
    
    while (polynomial.size()  > 1){
        if (polynomial.find("x")  < 10000) {
            // Parse the polynomial string.; find initial multiplicative factor.
            num = atof(polynomial.substr(0,polynomial.find("x")).c_str()); 

            // Removes everything to the first power
            polynomial.erase(0, polynomial.find("x") +2);

            // Finds the power
            power = atof(polynomial.substr(0,polynomial.find("^")).c_str()); 

            // Places number(s) into vector
            poly_double.push_back(num);
            poly_double.push_back(power);

            // erases first term
            polynomial.erase(0,polynomial.find("^") + 1);
        }
        else{
            offset = atof(polynomial.substr(0,polynomial.size()).c_str());
            polynomial.erase(0,2);
        }

    }

/*
    for (int il = 0; il < poly_double.size(); il++){
        cout << poly_double[il] << endl; 
    }
*/

    for (int ix = 0; ix < resolution; ix++){
        x = bound_a + (ix * (bound_b - bound_a)) / resolution;

        // finds height for later
        for (int il = 0; il < poly_double.size(); il +=2){
            height += pow(x, poly_double[il + 1]) * poly_double[il]; 
        }

        height += offset;

        for (int iy = 0; iy < resolution; iy++){
            y = bound_a + (iy * (bound_b - bound_a)) / resolution;
            if (y < height){
                count++;
            }
        }
    }

    integral = (count * ((bound_b - bound_a) / resolution) 
                      * ((bound_b - bound_a) / resolution));

    return integral;
}

