/*-----------igloo.cpp--------------------------------------------------------//
*
*            igloo.cpp
*
* Purpose: To answer fun questions with code.
*
* Concept: general concept provided in ../README. Be sure to read it, it's cool.
*
*   Notes: This is the first file I have written to solve this problem. I may
*          choose to solve it again with other programming languages in the 
*          near future, but for now... c++.
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include<vector>
#include<cstdlib>
#include<time.h>
#include<cmath>

using namespace std;

double integrate_sphere();
double integrate_monte(int resolution);
vector <double> populate();
vector <double> arrange(vector <double> people);
vector <double> find_height(double width);
int who_fits(vector <double> heights, vector<double> people);

int main(){

    // initialize random seed based on time
    srand (time(NULL));

    // Find peolep
    vector <double> people = populate();

    // organize people
    vector <double> organized_people = arrange(people);

    // figure out how much space each person could use in the igloo
    vector <double> igloo_h = find_height(0.1);
    
    // organize those heights
    vector <double> organized_heights = arrange(igloo_h);

    // squish everyone inside
    int saved_people = who_fits(organized_people, organized_heights);
    //cout << organized_people.size() << '\t' << saved_people << endl;

    // integrates a half-sphere with MONTE CARLO!
    double integral = integrate_monte(10000);
    cout << integral << endl;
    cout << "this is off by: " << (4.0 / 6.0) * M_PI - integral << endl;

}

// This function will integrate a half-sphere.
double integrate_sphere(){

    int count = 0, res = 1000;
    double integral, x, y, z;
    for (int ix = 0; ix < res; ix++){
        for (int iy = 0; iy < res; iy++){
            for (int iz = 0; iz < res / 2; iz++){
                x = (2 * ix / res) - 1; 
                y = (2 * iy / res) - 1;
                z = (2 * iz / res);
                if (((x*x) + (y*y) + (z*z)) < 1){
                    count += 1;
                }
            }
        }
    }

    integral = (count * (2.0 / 1000.0) * (2.0 / 1000.0) * (2.0 / 500.0));

    return integral;
}

// This function will integrate a half-sphere with MONTE CARLO INTEGRATION!!!
double integrate_monte(int resolution){

    double x, y, z, integral;
    int count = 0;
    for (int i = 0; i < resolution; i++){
        x = ((rand() % 10000) * (2.0 / 10000.0)) - 1;
        y = ((rand() % 10000) * (2.0 / 10000.0)) - 1;
        z = ((rand() % 10000) * (1.0 / 10000.0));
        if (x*x +y*y + z*z < 1){
            count++;
        }
    }

    // Scale factor is 4 because the range for (x,y,z) is (2,2,1)
    integral = ((double) count / (double) resolution) * 4;

    return integral;
}

// Finds the population of the village
vector <double> populate(){
    vector <double> population;
    int people_num = rand() % 100;

    for (int i = 0; i < people_num; i++){

        population.push_back((rand() % 1000) / 1000.0);
    }

    return population;
}

// Arranges the population in the village
vector <double> arrange(vector <double> people){

    vector <double> dummy = people;
    int chosen;
    double max;

    for (int id = 0; id< dummy.size(); id++){
        max = 0;
        for (int ip = 0; ip< people.size(); ip++){
            if (people[ip] > max){
                max = people[ip];
                chosen = ip;
            }
        }
        dummy[id] = max;
        people.erase(people.begin() + chosen);
    }
   
return dummy;
}

// Finds heights of igloo
vector <double> find_height(double width){

    vector<double> heights;
    int res = 2 / width;
    double x, y, height;

    for (int ix = 0; ix < res; ix++){
        for (int iy = 0; iy < res; iy++){
            x = (2.0 * ix / res) - 1;
            y = (2.0 * iy / res) - 1;
            if (x*x + y*y < 1){
                height = sqrt(1 - (x*x) - (y*y));
                heights.push_back(height);
            }
        }
    }
    return heights;
}

// Find how many people fit into igloo
int who_fits(vector <double> heights, vector<double> people){
    int fitting_people = 0;

    for (int ip = 0; ip < people.size(); ip++){
        if (people[ip] < heights[0]){
            fitting_people += 1;
            heights.erase(heights.begin());
        }
        
    }

    return fitting_people;
}

