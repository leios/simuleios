/*-------------euler.cpp------------------------------------------------------//
*
* Purpose: To implement a basic version of euler methods for random things
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include "../visualization/cairo/vec.h"
#include "../visualization/cairo/cairo_vis.h"

// Function to initialize the scene
std::vector<frame> init_scene(){
    vec res = {400, 300};
    int fps = 30;
    color bg_clr = {0,0,0,1};

    return init_layers(3, res, fps, bg_clr);
}

// Function to map coordinates
vec map_coordinates(vec pos, vec max, vec res){
    vec double_pos = {(pos.x / max.x) + 0.5*max.x,
                      (pos.y / max.y) + 0.5*max.y,
                      0};
    return {res.x * double_pos.x / max.x,
            res.y * double_pos.y / max.y, 0};
    
}

// Simple struct to hold a moving particle
struct Particle{
    vec pos, vel, acc;
};

vec find_acc(Particle &planet, Particle &ball){
    vec distance = planet.pos - ball.pos;
    vec unit_vector = {distance.x/length(distance), 
                       distance.y/length(distance),
                       distance.z/length(distance)};
    double diff = length(distance);
    double acc_mag = 1/(diff*diff);
    return acc_mag*distance;
}

// We will be returning the new position of the ball after moving forward 1 dt
void euler_ball(Particle &ball, double dt){
    ball.vel += ball.acc*dt;

    ball.pos += ball.vel*dt;
}

int main(){

    Particle planet, ball;

    planet.acc = {0,0,0};
    planet.vel = {0,0,0};
    planet.pos = {0,0,0};

    ball.acc = {0,0,0};
    ball.vel = {0,0,0};
    ball.pos = {5,0,0};

    double threshold = 0.1;
    double dt = 0.01;

    std::vector<frame> layers = init_scene();
    color white = {1,1,1,1};

    vec max = {5, 5, 0};
    vec res = {layers[0].res_x, layers[0].res_y, 0};
    vec pos = map_coordinates(planet.pos, max, res);
    grow_circle(layers[1], 1, pos, 50, white);

    pos = map_coordinates(ball.pos, max, res);
    grow_circle(layers[1], 1, pos, 10, white);

    while (distance(planet.pos, ball.pos) > threshold){
    //for(int i = 0; i < 100; ++i){
        ball.acc = find_acc(planet, ball);
        euler_ball(ball, dt);
        print(ball.acc);
        print(ball.vel);
        print(ball.pos);
        std::cout << distance(planet.pos, ball.pos) << '\n';
        std::cout << '\n';
    }

    draw_layers(layers);
}
