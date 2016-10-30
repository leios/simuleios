/*-------------viral_video.cpp------------------------------------------------//
*
* Purpose: To simulate how a video becomes viral
*
*   Notes: videos represented as integers, we don't care about their name
*          videos_seen will hold history for the viewer, we may cut this off
*          viewers have seen ~10 videos at the start of the simulation
*          the video we are propagating is video 0
*          Now, viewers may watch the same videos over gain, check affinity
*          Compile with "g++ viral_video.cpp"
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// Struct to hold all information for each individual viewer
struct viewer{
    std::vector<int> videos_seen;
    std::vector<double> affinities;
};

// Function to initialize all the people in the simulation
std::vector<viewer> init_viewers(int viewer_num, int num_videos);

// Function to find affinity with all other people 
std::vector<double> find_affinities(int individual,
                                    std::vector<viewer> &viewers);


/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main(){

    std::vector<viewer> viewers;
    viewers = init_viewers(100, 10);

    for (int i = 0; i < viewers.size(); i++){
        std::cout << i << '\t' << viewers[i].videos_seen[0] << '\t' 
                  << viewers[i].affinities[0] << '\n';
    }
}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Function to initialize all the people in the simulation
std::vector<viewer> init_viewers(int viewer_num, int num_videos){

    int history = 10;

    // Defining random distribution for videos seen by people
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int>
        chosen_videos_dist(1, num_videos);

    std::vector<viewer> viewers(viewer_num);

    // Defining videos seen by each person
    for (int i = 0; i < viewer_num; i++){
        viewers[i].videos_seen.reserve(history);
        for (int j = 0; j < history; j++){
            viewers[i].videos_seen.push_back(chosen_videos_dist(gen));
        }
    }

    // Denfining all the initial affinities
    for (int i = 0; i < viewer_num; i++){
        std::cout << i << '\n';
        viewers[i].affinities = find_affinities(i, viewers);
    }

    return viewers;
}

// Function to find affinity with all other people 
std::vector<double> find_affinities(int individual,
                                    std::vector<viewer> &viewers){

    std::vector<double> affinities(viewers.size());
    std::vector<int>::iterator it;
    double affinity;
    int shared = 0;

    // Searching through all the other viewers to find videos in common
    for (int i = 0; i < viewers.size(); i++){
        std::cout << i << '\n';
        if (individual != i){
            // Naive method to find all videos shared between to folks
            for (int j = 0; j < viewers[individual].videos_seen.size(); j++){
                it = find(viewers[i].videos_seen.begin(), 
                          viewers[i].videos_seen.end(),
                          viewers[individual].videos_seen[j]);
                if (it != viewers[i].videos_seen.end()){
                    shared++;
                }
            }
            affinity = (double)shared 
                       / (double)viewers[individual].videos_seen.size();
            affinities[i] = affinity;

            shared = 0;
        }
        else{
            affinities[i] = 0;
        }
    }

    return affinities;
}
