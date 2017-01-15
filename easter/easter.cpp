/*-------------easter.cpp-----------------------------------------------------//
*
* Purpose: To calculate the date of easter, based on Gauss's algotihm (~1816)
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include "../visualization/cairo/cairo_vis.h"

// Function to calculate the date of easter between start and finish
std::vector<int> easter_date_between(int start, int end);

// Function to calculate the date for a particular year
int easter_date(int year);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    // Creating scene and frames and stuff
    int fps = 30;
    double res_x = 400;
    double res_y = 400;
    vec res = {res_x, res_y};
    color bg_clr = {0, 0, 0, 1};
    color line_clr = {1, 1, 1, 1};

    std::vector<frame> layers = init_layers(3, res, fps, bg_clr);

    std::vector<int> dates = easter_date_between(1600, 2000);

    // Concatenate them all into a single array
    std::vector<int> dist(35);
    for (size_t i = 0; i < dates.size(); i++){
        dist[dates[i]]++;
        bar_graph(layers[1], 0, layers[1].curr_frame, layers[1].curr_frame+1,
                  dist, res_x, res_y, line_clr);
        layers[1].curr_frame++;
    }

    draw_layers(layers);
}

/*----------------------------------------------------------------------------//
* SUBROUTINES
*-----------------------------------------------------------------------------*/

// Function to calculate the date for a particular year
int easter_date(int year){
        int date;

        int a = year % 19;
        int b = year % 4;
        int c = year % 7;
        int k = year/100;
        int p = (13 + 8*k)/25;
        int q = k/4;
        int M = (15 - p + k - q) % 30;
        int N = (4 + k - q) % 7;
        int d = (19 * a + M) % 30;
        int e = (2*b + 4*c + 6*d + N) % 7;

        // determining whether the date is in March or April
        date = d+e;
/*
        if (22 + d + e < 31){
            date = 22 + d + e;
        }
        else{
            date = d+e;
        }

        if (d == 29 && e == 6){
            date = 19+9;
        } 

        
        if (d == 28 && e == 6 && (11 * M + 11) % 30 < 10){
            date = 18+9;
        } 
*/

        return date;

}

// Function to calculate the date of easter between start and finish
std::vector<int> easter_date_between(int start, int end){

    // Initialize the vector
    std::vector<int> dates(end - start+1);

    for (size_t i = 0; i < dates.size(); i++){
        dates[i] = easter_date(start + i);
    }

    return dates;
}
