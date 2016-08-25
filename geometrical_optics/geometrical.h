/*------------geometrical.h---------------------------------------------------//
*
* Purpose: Hold functions from geometrical.cpp
*
*-----------------------------------------------------------------------------*/

#ifndef GEOMETRICAL_H
#define GEOMETRICAL_H

#include<atomic>

/*----------------------------------------------------------------------------//
* STRUCTS / FUNCTIONS
*-----------------------------------------------------------------------------*/

// Spinlock class for parallelization
class spinlock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) { }
    }
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

// Constants
const int NUM_LIGHTS = 20;
const int TIME_RES = 500000;

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
static inline bool is_null(vec a) { return a.x == 0.0 && a.y == 0.0; }

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
template <typename T>
struct sphere{
    vec origin;
    double radius;
    double index_param;
    T index_fn;

    sphere(vec origin, double radius, double ip, const T &fn) 
          : origin(origin), radius(radius), index_param(ip), index_fn(fn){}

    vec normal_at(vec p) const{
        return normalize(p - origin);
    }

    bool contains(vec p) const {
        vec d = origin - p;
        return dot(d, d) < radius * radius;
    }

    double refractive_index_at (vec p) const {
        if (contains(p)){
            return index_fn(*this, p);
        }
        else{
            return 1.0;
        }
    }
};

// template function to create our sphere lens
template <typename T>
sphere<T> make_sphere(vec origin, double radius, double ip, const T& fn){
    return sphere<T>(origin, radius, ip, fn);
};

// Structs for all lenses
// Struct for simple lens
struct constant_index{
    //double index;
    //constant_index(double i) : index(i) {}
    double operator()(const sphere<constant_index>& lens, vec) const{
        return lens.index_param;
    }
};

// Struct for erf(1/r)
struct inverse_erf_index{
    double operator()(const sphere<inverse_erf_index>& lens, vec p) const{
        double r = (distance(lens.origin, p)) / lens.radius;
        return erf(lens.index_param / r);
    }
};

// Struct for the invisible lens
struct invisible_index{
    double operator()(const sphere<invisible_index>& lens, vec p) const {
        double index;
        double cutoff = 0.001;

        double a = lens.radius;
        double r = distance(lens.origin, p);
        if (r > cutoff){
            double q = cbrt(-(a/r) + sqrt((a * a) / (r * r) + 1.0 / 27.0));
            index = (q - 1.0 / (3.0 * q)) * (q - 1.0 / (3.0 * q));
        }
        else{
            r = cutoff;
            double q = cbrt(-(a/r) + sqrt((a * a) / (r * r) + 1.0 / 27.0));
            index = (q - 1.0 / (3.0 * q)) * (q - 1.0 / (3.0 * q));
        }

        return index;
    }
};

// Struct for piecemeal
struct piecemeal_index{
    double operator()(const sphere<piecemeal_index>& lens, vec p) const {
        double index;
        index = 1.5 - floor(distance(lens.origin, p) / lens.radius * 5) * 0.1;
        return index;
    }
};

// Struct for inverse (1/r)
struct inverse_index{
    double operator()(const sphere<inverse_index>& lens, vec p) const {
        return lens.index_param / (distance(lens.origin, p)/lens.radius);
    }
};

// Struct for r
struct r_index{
    double operator()(const sphere<r_index>& lens, vec p) const {
        return lens.index_param * (distance(lens.origin, p)/lens.radius);
    }
};

// Struct for sin_inverse_r_index = sin((1/param) * r) + 1.0
struct sin_inverse_r_index{
    double operator()(const sphere<sin_inverse_r_index>& lens, vec p) const {
        double r = distance(lens.origin, p);
        return (sin((.2/lens.index_param) * r)) * 0.5 + 2.0;
    }
};

// Struct for part of the batman function
struct batman_index{
    double operator()(const sphere<batman_index>& lens, vec p) const {
        double diff = distance(lens.origin, p) / lens.radius;
        double index = 0.5 * (fabs(0.5 * diff) + sqrt(1 - (pow(fabs(fabs(diff) - 2) - 1, 2))) -
                1 / 112 * (3 * sqrt(33) - 7) * diff * diff + 3 * sqrt(1 - (diff / 7) * (diff / 7) ) - 3) *
                ((diff + 4) / fabs(diff + 4) - (diff - 4) / fabs(diff - 4)) - 3 * sqrt(1 - (diff / 7) * (diff / 7));
        index = fabs(index);
        return index;
    }
};

// Struct for sigmoid function
struct sigmoid_index{
    double operator()(const sphere<sigmoid_index>& lens, vec p) const {
        double r = distance(lens.origin, p) / lens.radius;
        double index = 1.0 / (1.0 + exp(-lens.index_param *r));
        return index;
    }
};

// Struct for scale / cosh(r * index_param)
struct inverse_cosh_index{
    double operator()(const sphere<inverse_cosh_index>& lens, vec p) const {
        double r = distance(lens.origin, p) / lens.radius;
        double index = 1.0 / cosh(r * lens.index_param) + 1.0;
        return index;
    }
};

// Struct for erf_damped_sinusoid 
// exp(erf(sin(x) * cos(x) * exp((cos(x)* exp(sin(x))))))
struct erf_damped_sinusoid_index{
    double operator()(const sphere<erf_damped_sinusoid_index>& lens, vec p) 
        const {
        double x = distance(lens.origin, p) / lens.index_param;
        double index = exp(erf(sin(x) * cos(x) * exp((cos(x)* exp(sin(x))))));
        return index;
    }
};

// Struct for exp(erf(tan(x)))
struct exp_erf_tan_index{
    double operator()(const sphere<exp_erf_tan_index>& lens, vec p) const{
        double x = distance(lens.origin, p) / lens.index_param;
        double index = exp(erf(tan(x)));
        return index;
    }
};

// Struct for damped_sinusoid
// exp(sin(x) * cos(0.5 * x) * sin(pi / 7.0 + 5.0 * x)*cos(pi / 5.0 + 4.0 * x))
struct damped_sinusoid_index{
    double operator()(const sphere<damped_sinusoid_index>& lens, vec p) const{
        double x = distance(lens.origin, p) / lens.index_param;
        double index = exp(sin(x) * cos(0.5 * x) * sin(M_PI / 7.0 + 5.0 * x)*cos(M_PI / 5.0 + 2.0 * x)) ;
        return index;
    }
};

// Add overloads for 'normal_at' and 'refractive_index_at' for your own stuff,
// example (you'll need a separate struct for the different lenses):
//
// vec normal_at(const circle& lens, vec p) { ... }
// double refractive_index_at(const circle& lens, vec p) { ... }
bool inside_of(const simple& lens, vec p);
vec normal_at(const simple& lens, vec p);
double refractive_index_at(const simple& lens, vec p);

// Templated so it can accept any lens type. Stuff will dispatch at compile
// time, so the performance will be good
template <typename T>
ray_array light_gen(vec dim, const T& lens, double max_vel, double angle,
                    double offset);

// Same as above
template <typename T, typename I>
void propagate(I begin, I end, const T& lens,
               double step_size, double max_vel,
               frame &anim, double time);

// Template / Function for sweeping a single ray across the lens
template <typename T>
void propagate_sweep(const T& lens,
                     double step_size, double max_vel,
                     frame &anim);

// Template / function for a modified refractive index during propagation
template <typename T>
void propagate_mod(ray_array& rays, T& lens, double step_size,
                   double max_vel, frame &anim, double index_max);

// Function to write out index at any frame
void print_index(frame &anim, double index, color clr);

// Function to plot index function instead of lens and propagate
// Primarily for propagate_mod
template <typename T>
void draw_function(frame &anim, T& lens, double time, double index_max, 
                   double scale_y);

#endif
