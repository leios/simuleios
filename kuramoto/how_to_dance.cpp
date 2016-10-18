/*-------------how_to_dance.cpp-----------------------------------------------//
*
* Purpose: We are teaching Steve the Kuramoto oscillator how to dance with other
*          Kuramoto oscillators
*
*   Notes: This is a 1D simulation
*          compile with g++ how_to_dance.cpp
*          Note: Add in difference constants for difference dancers
*
*-----------------------------------------------------------------------------*/

#include "how_to_dance.h"
#include "how_to_dance_vis.h"

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/
int main(){

    // Initialization fo visualization
    frame anim;
    anim.create_frame(400,300,60, "/tmp/image");
    anim.init();
    anim.curr_frame = 1;

    create_bg(anim, 0,0,0);

    // Initialize group
    std::vector<oscillator> group = init_group(20, .2, 20.0);

    // Initializing steve
    oscillator steve = init_steve(0.25 * M_PI, 20.0);

    std::ofstream file("particle_output.dat", std::ofstream::out);

    synchronize(anim, group, steve, 0.1, .0000000001, file);

    anim.destroy_all();

    file.close();
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function to initialize group
std::vector<oscillator> init_group(int groupnum, double freq,
                                   double dance_floor){
    std::vector<oscillator> group(groupnum);

    // Define random distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        dance_floor_dist(-dance_floor * 0.5, dance_floor * 0.5);
    std::uniform_real_distribution<double>
        attractiveness_dist(0,1.0);

    for (int i = 0; i < groupnum; i++){
        group[i] = oscillator(dance_floor_dist(gen), 0.0, 0.0, 0.0, freq);
        group[i].ppos = group[i].pos;
        group[i].attr = attractiveness_dist(gen);
        //std::cout << group[i].pos << '\n';
    }

    return group;

}

// Function for initializing steve
oscillator init_steve(double phase, double max_freq){

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double>
        steve_dist(0, 2 * M_PI);
    oscillator steve = oscillator(0.0, 0.0, 0.0, steve_dist(gen), phase);
    return steve;

}

// Function to update phaseuency of steve and group
void update_phase(std::vector<oscillator> &group, oscillator &steve,
                      double dt){
    // group first
    for (size_t i = 0; i < group.size(); i++){
        group[i].phase += group[i].freq * dt;
    }

    // now for ... steve
    steve.phase += steve.freq * dt;
}

// Function to synchronize steve to the rest of the group
// Note: we can weight the dancers based on their proximity to steve
void synchronize(frame &anim, std::vector<oscillator> &group, oscillator &steve,
                 double dt, double cutoff, std::ofstream &file){

    double sum;
    double nat_freq = steve.freq;
    double cairo_pos, cairo_steve_pos;
    color white = {1, 1, 1};
    color red = {1, 0, 0};
    std::cout << "group frequency is: " << group[0].freq << '\n';

    // Now we need to synchronize steve to the group
    // Note: everyone is oscillating at the same phaseuency except for steve
    std::cout << steve.freq - group[0].freq << '\n';
    //while ((steve.freq - group[0].freq) > 0.01){
    for (int i = 0; i < 500; i++){
        update_phase(group, steve, dt);

        // Note that all frequencies of members in the group are the same
        if (i > 100){
            sum = sin(group[0].phase - steve.phase);
            steve.freq = nat_freq + sum;
        }

        // Update positions for members in the group
        find_acc(group, steve, cutoff);
        verlet(group, dt);
        //output_pos(group, steve, file);
        // output steve's position
        cairo_steve_pos = (steve.pos + 10) * anim.res_x / 20.0;
        draw_human(anim, cairo_steve_pos, steve.phase, steve.freq, red);
        for (auto& member : group){
            cairo_pos = (member.pos + 10) * anim.res_x / 20.0 ;
            draw_human(anim, cairo_pos, member.phase, member.freq, white);
        }
        anim.curr_frame++;

        //std::cout << sum << '\t' << steve.freq - group[0].freq << '\n';
    }


    anim.draw_frames();
}

// Function to calculate change in position
void verlet(std::vector<oscillator> &group, double dt){

    // changing the position of all dancers in simulation
    for (auto& member : group){
        //std::cout << member.acc << '\n';
        double temp_x = member.pos;
        member.pos = 2 * member.pos - member.ppos + member.acc * dt*dt;
        member.ppos = temp_x;
        if (member.pos > 10){
            member.pos = 10;
        }
        if (member.pos < -10){
            member.pos = -10;
        }

        member.vel = (member.pos - member.ppos) / (2 * dt);
    }
}

// Find acceleration of all dancers on the dancefloor
void find_acc(std::vector<oscillator> &group, oscillator &steve, double cutoff){

    double x_diff, freq_diff;

    // checking how far off steve is
    for (auto& member : group){
        x_diff = member.pos - steve.pos;
        freq_diff = member.freq - steve.freq;
        // Repulsive force
        if ((freq_diff*freq_diff) > cutoff){
            member.acc = member.attr * 0.1 / (x_diff);
        }
        // Attractive force
        else{
            member.acc = - member.attr * 0.1 * x_diff - member.vel * 0.5;
        }

        //std::cout << member.acc << '\n';
    }
}

// Function to output all positions
void output_pos(std::vector<oscillator> &group, oscillator &steve, 
                std::ofstream &file){

    std::vector<std::string> filenames(group.size());

    for (int i = 0; i < group.size(); i++){
        filenames[i] = "file" + std::to_string(i) + ".dat";
    }

    std::vector<std::ofstream> output_files;
    output_files.reserve(group.size());

    for (int i = 0; i < group.size(); i++){
        output_files.emplace_back(filenames[i], std::ofstream::app);
    }

    for (int i = 0; i < group.size(); i++){
        output_files[i] << group[i].pos << '\n';
    }

/*
    // loop for closing everything
    for (auto& file : output_files){
        file.close();
    }
*/

    // Output steve first
    file << steve.pos << '\t' << steve.phase << '\n';

    // Outputting all other people
    for (auto& member : group){
        file << member.pos << '\t' << member.phase << '\n';
    }

    file << '\n' << '\n';
}

// Function to return sign of double
double sign(double variable){
    if (variable < 0){
        return -1.0;
    }
    else if (variable == 0){
        return 0.0;
    }
    else{
        return 1.0;
    }
}
