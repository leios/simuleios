/*-------------euler.cpp------------------------------------------------------//
*
* Purpose: To implement a basic version of euler methods for random things
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include "../visualization/cairo/vec.h"
#include "../visualization/cairo/cairo_vis.h"

// Simple struct to hold a moving particle
struct Particle{
    vec pos, vel, acc;
};

// Function to initialize the scene
std::vector<frame> init_scene(){
    vec res = {600, 400};
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

void verlet_ball(Particle &ball, vec acc_prev, double dt){
    ball.pos += ball.vel*dt + 0.5*ball.acc*dt*dt;
    ball.vel += 0.5*(acc_prev + ball.acc)*dt;
}

void visualize_euler_ball(){

    Particle planet;

    planet.acc = {0,0,0};
    planet.vel = {0,0,0};
    planet.pos = {0,0,0};

    std::vector<Particle> balls(5), verlet_balls(5);
    for (Particle &ball : balls){
        ball.acc = {0,0,0};
        ball.vel = {0,1,0};
        ball.pos = {5,0,0};
    }

    for (Particle &ball : verlet_balls){
        ball.acc = {0,0,0};
        ball.vel = {0,1,0};
        ball.pos = {5,0,0};
    }

    balls[1].vel = {1,0,0};
    balls[1].pos = {0,5,0};
    verlet_balls[1].vel = {1,0,0};
    verlet_balls[1].pos = {0,5,0};

    balls[2].vel = {0.5,-0.5,0};
    balls[2].pos = {4,4,0};
    verlet_balls[2].vel = {0.5,-0.5,0};
    verlet_balls[2].pos = {4,4,0};

    balls[3].vel = {1.5,0,0};
    balls[3].pos = {0,5,0};
    verlet_balls[3].vel = {1.5,0,0};
    verlet_balls[3].pos = {0,5,0};

    balls[4].vel = {-0.5,0.5,0};
    balls[4].pos = {-4,-4,0};
    verlet_balls[4].vel = {-0.5,0.5,0};
    verlet_balls[4].pos = {-4,-4,0};

    double threshold = 0.1;
    double dt = 0.1;

    std::vector<frame> layers = init_scene();
    color white = {1,1,1,1};
    color blue = {0.25,0.25,1,1};
    color pink = {1,0,1,1};

    vec max = {5, 5, 0};
    vec res = {(double)layers[0].res_x, (double)layers[0].res_y, 0};
    vec pos = map_coordinates(planet.pos, max, res);
    grow_circle(layers[1], 1, pos, 50, white);

    for (Particle &ball : balls){
        pos = map_coordinates(ball.pos, max, res);
        grow_circle(layers[1], 1, 100, 200, pos, 10, blue);
    }

    //while (distance(planet.pos, ball.pos) > threshold){
    int timesteps = 500;
    for(int i = 0; i < timesteps; ++i){
        if (i > timesteps/2){
            for (Particle &ball : verlet_balls){
                vec acc_prev = ball.acc;
                ball.acc = find_acc(planet, ball);
                verlet_ball(ball, acc_prev, dt);
                vec mapped_coord = map_coordinates(ball.pos, max, res);
                draw_filled_circle(layers[1], mapped_coord, 10, 200 + i, pink);
            }
        }
        for (int j = 0; j < balls.size(); ++j){
            balls[j].acc = find_acc(planet, balls[j]);
            euler_ball(balls[j], dt);
            vec mapped_coord = map_coordinates(balls[j].pos, max, res);
            draw_filled_circle(layers[1], mapped_coord, 10, 200 + i, blue);
            verlet_balls[j] = balls[j];
        }
    }

    draw_layers(layers);
}

std::vector<double> solve_euler(double step, double xmax){

    int n = (int)(xmax / step);
    std::vector<double> euler_output(n);

    //euler_output[0] = 0;
    euler_output[0] = 1;
    for (int i = 1; i < n; ++i){
        double xval = i*step;

        euler_output[i] = euler_output[i-1] +xval*step;
        //euler_output[i] = (-2.3*exp(xval + step) -euler_output[i-1])/step;
    }

    return euler_output;
}

// Function to visualize euler instability
void visualize_instability(){
    std::vector<frame> layers = init_scene();
    int n = 100;
    double xmax = 10;
    std::vector<double> array(n);
    for (int i = 0; i < n; ++i){
        array[i] = pow((i*xmax/n),2)/2;
        //array[i] = exp(-2.3*(i*xmax/n));
    }

    std::vector<double> euler_array = solve_euler(0.5, xmax);
    std::vector<double> euler_array_2 = solve_euler(0.1, xmax);
    //std::vector<double> euler_array = solve_euler(0.1, xmax);
    //std::vector<double> euler_array_2 = solve_euler(2, xmax);

    for (double &val : euler_array){
        std::cout << val << '\n';
    }

    vec ori = {layers[0].res_x*0.5,layers[0].res_y*0.5,0};
    vec dim = {300,200,0};
    color bar_color = {1, 1, 1, 1};
    color plot_color = {0.5, 0.5, 1, 1};
    color plot_color_2 = {1, 0.5, 0.5, 1};
    color plot_color_3 = {0.5, 1, 0.5, 1};
    plot(layers[1], array, 10, 0, 500, ori, dim, bar_color, plot_color);
    plot(layers[2], euler_array, 10, 0, 500, ori, dim, bar_color, plot_color_2);
    plot(layers[2], euler_array_2, 10, 0, 500, ori, 
         dim, bar_color, plot_color_3);

    draw_layers(layers);
}

int main(){
    //visualize_euler_ball();
    visualize_instability();
}
