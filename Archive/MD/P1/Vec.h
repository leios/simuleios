#ifndef VEC_H

#include <cmath>
namespace sim{
    /* Vec is a mathmatical 3-dimensional xyz vector.
       It avoids tripply redundant code by allowing to multiply Vecs
       which multiplies their respective xyz values */
    struct Vec{
        //to constuct a Vec you need to give it the xyz values
        Vec(double x, double y, double z) : x(x), y(y), z(z){}
        //sum of the vector
        double sum() const{
            return x + y + z;
        }
        //product of the vector
        double prod() const{
            return x * y * z;
        }
        //length of the vector
        double length() const{
            return sqrt(x * x + y * y + z * z);
        }
        //+=. -=, *=, /= operators for Vec-Vec and Vec-double
        Vec& operator +=(const Vec& other){
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }
        Vec& operator +=(const double other){
            x += other;
            y += other;
            z += other;
            return *this;
        }
        Vec& operator -=(const Vec& other){
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }
        Vec& operator -=(const double other){
            x -= other;
            y -= other;
            z -= other;
            return *this;
        }
        Vec& operator *=(const Vec& other){
            x *= other.x;
            y *= other.y;
            z *= other.z;
            return *this;
        }
        Vec& operator *=(const double other){
            x *= other;
            y *= other;
            z *= other;
            return *this;
        }
        Vec& operator /=(const Vec& other){
            x /= other.x;
            y /= other.y;
            z /= other.z;
            return *this;
        }
        Vec& operator /=(const double other){
            x /= other;
            y /= other;
            z /= other;
            return *this;
        }

        double x, y, z;
    };

    //operators +, -, *, / for Vec-Vec, Vec-double and double-Vec
    Vec operator+(const Vec& lhs, const Vec& rhs){
        return Vec(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
    }
    Vec operator*(const Vec& lhs, const Vec& rhs){
        return Vec(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
    }
    Vec operator-(const Vec& lhs, const Vec& rhs){
        return Vec(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    }
    Vec operator/(const Vec& lhs, const Vec& rhs){
        return Vec(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
    }

    Vec operator+(const Vec& lhs, const double rhs){
        return Vec(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
    }
    Vec operator+(const double lhs, const Vec& rhs){
        return Vec(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
    }
    Vec operator-(const Vec& lhs, const double rhs){
        return Vec(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
    }
    Vec operator-(const double lhs, const Vec& rhs){
        return Vec(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
    }
    Vec operator*(const Vec& lhs, const double rhs){
        return Vec(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
    }
    Vec operator*(const double lhs, const Vec& rhs){
        return Vec(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
    }
    Vec operator/(const Vec& lhs, const double rhs){
        return Vec(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
    }
    Vec operator/(const double lhs, const Vec& rhs){
        return Vec(lhs / rhs.x, lhs / rhs.y, lhs / rhs.z);
    }
}

#endif //VEC_H
