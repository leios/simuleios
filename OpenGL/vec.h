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

// Struct to hold indices for triangle
struct triangle{
    // ab, bc, ca are from geometric representations of trangle with 3 sides
    // a, b, and c
    int ab, bc, ca;
    triangle() : ab(0), bc(0), ca(0) {};
    triangle(int p1, int p2, int p3) : ab(p1), bc(p2), ca(p3) {};
};

// Struct for 3d points, stored as floats
struct vec {
    float x, y, z;

    vec() : x(0.0), y(0.0), z(0.0) {}
    vec(float x0, float y0, float z0) : x(x0), y(y0), z(z0) {}

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

    vec& operator*=(float rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    vec& operator/=(float rhs) {
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

inline vec operator*(vec lhs, float b) {
    return vec(lhs.x * b, lhs.y * b, lhs.z * b);
}

inline vec operator*(float lhs, vec rhs) {
    return vec(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
}

inline vec operator/(vec lhs, float rhs) {
    auto inv = 1.0 / rhs;
    return lhs * inv;
}

inline float dot(vec lhs, vec rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline vec cross(vec lhs, vec rhs) {
    return vec(lhs.y * rhs.z - lhs.z * rhs.y,
               lhs.z * rhs.x - lhs.x * rhs.z,
               lhs.x * rhs.y - lhs.y * rhs.x);
}

inline float length(vec lhs) {
    return sqrt(dot(lhs, lhs));
}

inline vec normalize(vec lhs) {
    return lhs / length(lhs);
}

inline float distance(vec lhs, vec rhs) {
    return length(lhs - rhs);
}

inline void print(vec v) {
    printf("% .3lf, % .3lf, % .3lf\n", v.x, v.y, v.z);
}

#endif
