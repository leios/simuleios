/*-------------geometrical_optics.cpp-----------------------------------------//
*
*              geometrical optics
*
* Purpose: to simulate light going through a lens with a variable refractive
*          index. Not wave-like.
*
*-----------------------------------------------------------------------------*/

#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

/*----------------------------------------------------------------------------//
* STRUCTS / FUNCTIONS
*-----------------------------------------------------------------------------*/

// Constants
const int NUM_LIGHTS = 10;
const int TIME_RES = 250;

// A very simple vector type with operators that are used in this file
struct vec {
    double x, y;

    vec() : x(0.0), y(0.0) {}
    vec(double x0, double y0) : x(x0), y(y0) {}
};

// The static inlines are our best bet to force the inlining of the functions
// without using platform-specific extensions
static inline vec& operator+=(vec& a, vec b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

static inline vec operator-(vec a) { return vec(-a.x, -a.y); }
static inline vec operator+(vec a, vec b) { return vec(a.x + b.x, a.y + b.y); }
static inline vec operator-(vec a, vec b) { return vec(a.x - b.x, a.y - b.y); }
static inline vec operator*(vec a, double b) { return vec(a.x * b, a.y * b); }
static inline vec operator*(double a, vec b) { return b * a; }

static inline vec operator/(vec a, double b) {
    double inv = 1.0 / b;
    return a * inv;
}

static inline double dot(vec a, vec b) { return a.x * b.x + a.y * b.y; }
static inline double length(vec a) { return sqrt(dot(a, a)); }
static inline vec normalize(vec a) { return a / length(a); }
static inline double distance(vec a, vec b) { return length(a - b); }

// This normally wouldn't store the previous index, but there's no
// ray-shape intersection code yet.
struct ray {
    vec p, v;
    double previous_index;
};

// A convenience shorthand so we don't have to write the full type everywhere
using ray_array = std::array<ray, NUM_LIGHTS>;

// A struct describing a simple lens. Add additional lenses by adding
// a new struct and overloading the corresponding functions (see below)
struct simple {
    double left, right;
    simple(double l, double r) : left(l), right(r) {}
};

// A simple struct for circular / spherical lens
struct sphere{
    double radius;
    vec origin;
    sphere(double rad, double x, double y) : radius(rad), origin(x, y) {}
};

// Add overloads for 'normal_at' and 'refractive_index_at' for your own stuff,
// example (you'll need a separate struct for the different lenses):
//
// vec normal_at(const circle& lens, vec p) { ... }
// double refractive_index_at(const circle& lens, vec p) { ... }
bool inside_of(const simple& lens, vec p);
bool inside_of(const sphere& lens, vec p);
vec normal_at(const simple& lens, vec p);
vec normal_at(const sphere& lens, vec p);
double refractive_index_at(const simple& lens, vec p);
double refractive_index_at(const sphere& lens, vec p);

// Templated so it can accept any lens type. Stuff will dispatch at compile
// time, so the performance will be good
template <typename T>
ray_array light_gen(vec dim, const T& lens, double max_vel, double angle);

// Same as above
template <typename T>
void propagate(ray_array& rays, const T& lens,
               double step_size, double max_vel,
               std::ofstream &output);

/*----------------------------------------------------------------------------//
* MAIN
*-----------------------------------------------------------------------------*/

int main() {
    // defines output
    std::ofstream output("out.dat", std::ofstream::out);

    vec dim = {4, 10};
    double max_vel = 1;

    // Implement other lenses and change this line to use them
    sphere lens = {4, 5, 5};
    ray_array rays = light_gen(dim, lens, max_vel, 0 /*0.523598776*/);
    propagate(rays, lens, 0.1, max_vel, output);

    output << "\n \n5	0	0	0 \n5	5	0	0 \n";

}

/*----------------------------------------------------------------------------//
* SUBROUTINE
*-----------------------------------------------------------------------------*/

// Refracts the given normalized vector "l", based on the normalized normal "n"
// and the given index of refraction "ior", where ior = n1 / n2
vec refract(vec l, vec n, double ior) {
    double c = dot(-n, l);

    std::cout << c << '\t' << n.x << '\t' << n.y << '\t' 
              << l.x << '\t' << l.y << '\n';

    // If the normal points towards the wrong side (with no light) then
    // negate it and start again
    if (c < 0.0)
        return refract(l, -n, ior);

    return ior * l + (ior * c - sqrt(1.0 - ior * ior * (1.0 - c * c))) * n;
}

template <typename T>
ray_array light_gen(vec dim, const T& lens, double max_vel, double angle) {
    ray_array rays;
    vec velocity = vec(cos(angle), sin(angle)) * max_vel;

    // Create rays
    for (size_t i = 0; i < rays.size(); i++) {
        rays[i].p = vec(0.0, 3 + i * dim.x / NUM_LIGHTS);
        rays[i].v = velocity;
        rays[i].previous_index = refractive_index_at(lens, rays[i].p);
    }

    return rays;
}

template <typename T>
void propagate(ray_array& rays, const T& lens,
               double step_size, double max_vel,
               std::ofstream& output) {

    // move simulation every timestep
    for (auto& ray : rays) {
        for (size_t i = 0; i < TIME_RES; i+= 1){
            ray.p += ray.v * step_size;

            double n1 = ray.previous_index;
            double n2 = refractive_index_at(lens, ray.p);

            // If the ray passed through a refraction index change
            if (n1 != n2) {
                vec n = normal_at(lens, ray.p);
                vec l = normalize(ray.v);
                double ior = n1 / n2;

                vec refracted = refract(l, n, ior);
                // Multiply with ior * length(ray.v) to get the proper velocity
                // for the refracted vector
                ray.v = normalize(refracted) * ior * length(ray.v);
            }

            ray.previous_index = n2;

            output << ray.p.x <<'\t'<< ray.p.y << '\t'
                   << ray.v.x <<'\t'<< ray.v.y << '\n';
        }
        output << '\n' << '\n';
    }

}

// Inside_of functions
// simple lens slab
bool inside_of(const simple& lens, vec p) {
    return p.x > lens.left && p.x < lens.right;
}

// Circle / sphere
bool inside_of(const sphere& lens, vec p) {
    double diff = distance(lens.origin, p);
    return diff < lens.radius;
}

// Find the normal
// Lens slab
vec normal_at(const simple&, vec) {
    return normalize(vec(-1.0, 0.0));
}

// Circle / sphere
// ERROR: This is defined incorrectly!
vec normal_at(const sphere& lens, vec p) {
    //return normalize(vec(-1.0, 0.0));
    return normalize(p - lens.origin);
}

// find refractive index
// Lens slab
double refractive_index_at(const simple& lens, vec p) {
    return inside_of(lens, p) ? 1.4 : 1.0;
}

// Circle / sphere
double refractive_index_at(const sphere& lens, vec p) {
    //return inside_of(lens, p) ? 1.4 : 1.0;

    double index, diff;

    if (inside_of(lens, p)){
        diff = distance(lens.origin, p);
        index = 1.0 / diff;
    }
    else{
        index = 1;
    }
    
    return index;

}
