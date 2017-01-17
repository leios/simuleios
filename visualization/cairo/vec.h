/*------------vec.h-----------------------------------------------------------//
*
* Purpose: hold all vector operations for Barnes Hut
*
*   Notes: This file was written by Gustorn
*
*-----------------------------------------------------------------------------*/

#ifndef VEC_H
#define VEC_H

#include <cmath>
#include <cstdio>

struct vec {
    double x, y, z;

    vec() : x(0.0), y(0.0), z(0.0) {}
    vec(double x0, double y0) : x(x0), y(y0), z(0.0) {}
    vec(double x0, double y0, double z0) : x(x0), y(y0), z(z0) {}

    vec& operator+=(vec rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    vec& operator-=(vec rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    vec& operator*=(double rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    vec& operator/=(double rhs) {
        auto inv = 1.0 / rhs;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }
};

inline vec operator-(vec lhs) {
    return vec(-lhs.x, -lhs.y, -lhs.z);
}

inline vec operator+(vec lhs, vec rhs) {
    return vec(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

inline vec operator-(vec lhs, vec rhs) {
    return vec(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

inline vec operator*(vec lhs, double b) {
    return vec(lhs.x * b, lhs.y * b, lhs.z * b);
}

inline vec operator*(double lhs, vec rhs) {
    return vec(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

inline vec operator/(vec lhs, double rhs) {
    auto inv = 1.0 / rhs;
    return lhs * inv;
}

inline double dot(vec lhs, vec rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline vec cross(vec lhs, vec rhs) {
    return vec(lhs.y * rhs.z - lhs.z * rhs.y,
               lhs.z * rhs.x - lhs.x * rhs.z,
               lhs.x * rhs.y - lhs.y * rhs.x);
}

inline double length(vec lhs) {
    return sqrt(dot(lhs, lhs));
}

inline vec normalize(vec lhs) {
    return lhs / length(lhs);
}

inline double distance(vec lhs, vec rhs) {
    return length(lhs - rhs);
}

inline void print(vec v) {
    printf("% .3lf, % .3lf, % .3lf\n", v.x, v.y, v.z);
}

#endif
