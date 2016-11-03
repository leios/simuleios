/*-------------viral_video.cpp------------------------------------------------//
*
* Purpose: To simulate how a video becomes viral
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include "../visualization/cairo/cairo_vis.h"

// Struct for videos
struct video{
    int id;
    double val = 1.0;
};

// equality operator for videos
bool operator==(const video& a, const video& b){
    return (a.id == b.id);
}

// Struct to hold all information for each individual viewer
struct viewer{
    std::vector<video> video_list;
    //std::vector<video> videos_seen;
    std::unordered_map<int, double> videos_seen;
    std::vector<double> affinities;
    int next_video = -1;
    int curr_video = -1;
    bool has_seen_0 = 0;
    int index;

    // These are positional arguments for visualization and movement simulation
    // p and pp are position and previous position respectively
    vec p, pp, vel, acc;
};

// Function to initialize all the people in the simulation
std::vector<viewer> init_viewers(frame &anim, int viewer_num, int num_videos, 
                                 int subscribers);

// Function to find affinity with all other people 
std::vector<double> find_affinities(int individual,
                                    std::vector<viewer> &viewers);

// Function to provide videos to all viewers
void provide_videos(std::vector<viewer> &viewers, int cutoff, int num_videos);

// Function for choosing videos
void choose_video(std::vector<viewer> &viewers);

// Function to watch each individual's next video
void watch_video(std::vector<viewer> &viewers);

// Function to share videos with other people randomly in the network
// Note that this simply forces the individuals to watch a particular video 
// next timestep
void share(std::vector<viewer> &viewers, std::vector<int> &network, 
           int video_id);

// Function to output which videos are being watched to a file.
void output_viewers(std::vector<viewer> &viewers, std::ofstream &file);

// Function to place viewers in a frame for visualization
void draw_viewers(frame &anim, std::vector<viewer> &viewers);
