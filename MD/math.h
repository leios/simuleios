#ifndef MD_MATH_H
#define MD_MATH_H

#include <cmath>
#include <ostream>

namespace sim {

inline double square(double value) {
    return value * value;
}

struct Point;

struct Vector {
    double x, y, z;

    Vector() : Vector{0.0, 0.0, 0.0} {}
    Vector(double x0, double y0, double z0) : x{x0}, y{y0}, z{z0} {}
    explicit Vector(const Point&);

    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vector& operator+=(const Vector& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Vector& operator-=(const Vector& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    Vector& operator*=(double rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    Vector& operator/=(double rhs) {
        x /= rhs;
        y /= rhs;
        z /= rhs;
        return *this;
    }
};

struct Point {
    double x, y, z;

    Point() : Point{0.0, 0.0, 0.0} {}
    Point(double x0, double y0, double z0) : x{x0}, y{y0}, z{z0} {}
    explicit Point(const Vector& v) : Point{v.x, v.y, v.z} {}

    Point& operator+=(const Vector& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    Point& operator-=(const Vector& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }
};

inline Vector::Vector(const Point& p) : Vector{p.x, p.y, p.z} {}

inline Vector operator+(const Vector& lhs, const Vector& rhs) {
    return Vector{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline Vector operator-(const Vector& lhs, const Vector& rhs) {
    return Vector{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vector operator*(const Vector& lhs, double rhs) {
    return Vector{lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
}

inline Vector operator*(double lhs, const Vector& rhs) {
    return rhs * lhs;
}

inline Vector operator/(const Vector& lhs, double rhs) {
    return Vector{lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
}

inline Point operator+(const Point& lhs, const Vector& rhs) {
    return Point{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline Point operator-(const Point& lhs, const Vector& rhs) {
    return Point{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vector operator-(const Point& lhs, const Point& rhs) {
    return Vector{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

// A very simple template function, what this basicly means is take 2 different
// types (T, U) and if they provide an interface (have x, y, z members) then do
// the operation defined. This is here so there's no need to do all the
// combinations for Vector and Point
template <typename T, typename U>
double dot(const T& lhs, const U& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline std::ostream& operator<<(std::ostream& out, const Vector& rhs) {
    return out << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
}

inline std::ostream& operator<<(std::ostream& out, const Point& rhs) {
    return out << "(" << rhs.x << ", " << rhs.y << ", " << rhs.z << ")";
}

template <typename T>
T uniform_in_bounds2d(double from, double to) {
    static std::mt19937 engine{std::random_device{}()};

    // Creating distributions is really easy, and makes the code more readable
    auto distribution = std::uniform_real_distribution<double>{from, to};

    return T{distribution(engine), distribution(engine), 0.0};
}


}

#endif
