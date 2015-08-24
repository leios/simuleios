#include <array>
#include <iomanip>
#include <ostream>

// The template part means that we can create square matrices with arbitrary sizes.
// Since we're using std::array for storage things will be allocated on the
// stack, so you shouldn't use sizes that are too big (regular arrays, like
// double matrix[] have the same limitation, it's not unique to this custom
// class).
//
// A small foreword: any function inside a struct (without the static keyword)
// takes an implicit 'this' parameter, meaning it automatically has access
// to the variables in that instance. An example:
//
// ----------------------------------
// std::vector<int> test = {1, 2, 3};
// std::vector<int> other = {1, 2};
// test.size();
// other.size();
// ----------------------------------
//
// Here size() is defined as a function inside the class, so it knows what the
// current size of the vector it's been called on is. So for test it can return
// 3, while for other it can return 2.
//
// Member functions can be roughly translated in the following manner (you cannot
// actually do this in code, but it's a useful way to think about it):
//
// ------------------------------------------------------
// instance.function(params) = function(instance, params)
// ------------------------------------------------------
//
//
// An other gotcha (and the reason I had two operator[] functions in my matrix example
// on stream) is that some functions can modify the object and some cannot. So if you have
// a member function declaration that ends with 'const', you know that (under normal
// circumstances) it will never modify the object:
//
// ------------------------------------------------------
// struct test {
//     int a = 10;
//
//     void print() const {...}
//     void mutate() {...}
// };
// ------------------------------------------------------
//
// Here 'print' is only allowed to read from the member variables and call other functions
// marked as const (or functions that take const matrix&), while mutate is allowed to
// read and modify the member variables and call funcions that take either matrix& or
// const matrix&. This can reduce bugs considerably if you annotate your methods correcty.
//
// Now onto the actual matrix class
template <size_t dim>
class matrix {
public:
    static_assert(dim >= 2, "The smallest supported matrix<size> is 2x2");

    // A convenient way to access the number of rows and columns of the
    // matrix
    static const size_t size = dim;

    // Creates a matrix with all 0s as the initial value
    matrix() : data{0} {}

    // This allows us to construct a matrix with arbitrary starting values.
    // This is a moderately advanced feature, read up on variadic templates if you
    // are interested.
    template <typename... Args>
    matrix(Args... values) : data{static_cast<double>(values)...} {}

    // We access an element of the matrix with the following syntax:
    // element at row 1, column 3 = matrix(1, 3).
    //
    // This allows us to index the array without leaking any of the
    // abstractions to the outside world
    //
    // The other way is to copy the two operator[] functions that I showed you
    // on stream and replace the next two functions with them. If you're really
    // used to the matrix[row][col] syntax, then feel free, but this is a better
    // way of doing it (big linalg libraries use this type as well)
    double operator()(size_t row, size_t col) const {
        return data[row][col];
    }

    // This one is allowed to mutate the matrix, so something like this is allowed:
    // matrix(1, 3) = 5;
    // This will set the value at row 1, column 3 to 5
    double& operator()(size_t row, size_t col) {
        return data[row][col];
    }

    // Next is arithmetic assignment operators. These are more efficient than the
    // regular ones, and can be more convenient to use. By convention a matrix& is
    // returned to allow chaining assigments
    matrix& operator+=(const matrix& rhs) {
        for (size_t row = 0; row < size; ++row) {
            for (size_t col = 0; col < size; ++col) {
                data[row][col] += rhs.data[row][col];
            }
        }
        // The implicit this parameter I was talking about. It's a pointer, hence the
        // dereferencing
        return *this;
    }

    matrix& operator-=(const matrix& rhs) {
        for (size_t row = 0; row < size; ++row) {
            for (size_t col = 0; col < size; ++col) {
                data[row][col] -= rhs.data[row][col];
            }
        }
        return *this;
    }

    matrix& operator*=(double rhs) {
        for (size_t row = 0; row < size; ++row) {
            for (size_t col = 0; col < size; ++col) {
                data[row][col] *= rhs;
            }
        }
        return *this;
    }

    matrix& operator/=(double rhs) {
        // double division is slow, multiplication is much faster
        double inverse = 1.0 / rhs;

        for (size_t row = 0; row < size; ++row) {
            for (size_t col = 0; col < size; ++col) {
                data[row][col] *= inverse;
            }
        }
        return *this;
    }

private:
    std::array<std::array<double, size>, size> data;
};

template <size_t size>
matrix<size> identity() {
    matrix<size> ret;
    for (size_t i = 0; i < size; ++i) {
        ret(i,i) = 1.0;
    }
    return ret;
}

// Since NxN determinants are not that trivial to implement and it doesn't help to
// demonstrate the point of this implementation, I'll skip it. It can be done for arbitrary
// sizes with a divide & conquer recursive approach, which could also be parallelized for
// better performance

// 2x2 matrix<size> determinant
inline double det(const matrix<2>& m) {
    return m(0,0) * m(1,1) - m(0,1) * m(1,0);
}

// 3x3 matrix<size> determinant
inline double det(const matrix<3>& m) {
    return m(0,0) * (m(1,1) * m(2,2) - m(1,2) * m(2,1)) -
           m(0,1) * (m(1,0) * m(2,2) - m(1,2) * m(2,0)) +
           m(0,2) * (m(1,0) * m(2,1) - m(1,1) * m(2,0));
}

// Trace is easily generalized over matrix size
template <size_t size>
double trace(const matrix<size>& m) {
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += m(i,i);
    }
    return sum;
}

// There will be a common pattern here. We are reusing the results from our
// arithmetic assignment operators, so we don't have to repeat any logic that
// we could possibly screw up. This optimizes out to perfect assembly. One important
// note on this:
//
// ########
// ADVANCED
// ########
// We are taking advantage of a compiler feature called (Named) Return Value Optimization.
// I declared the return value in the (1)first line of the function, mutated it during the function,
// then returned it by the (2)only return statement in the function.
// If (1) and (2) are met you can rely on NRVO to optimize the function in a way that there are
// no unnecessary copies made.
//
// In this function we copy every double in lhs and rhs exactly once and add them together. This is
// the optimal solution
template <size_t size>
matrix<size> operator+(const matrix<size>& lhs, const matrix<size>& rhs) {
    matrix<size> ret{lhs};
    ret += rhs;
    return ret;
}

template <size_t size>
matrix<size> operator-(const matrix<size>& lhs, const matrix<size>& rhs) {
    matrix<size> ret{lhs};
    ret -= rhs;
    return ret;
}

// Right scalar multiplication
template <size_t size>
matrix<size> operator*(const matrix<size>& lhs, double rhs) {
    matrix<size> ret{lhs};
    ret *= rhs;
    return ret;
}

// Left scalar multiplication
template <size_t size>
matrix<size> operator*(double lhs, const matrix<size>& rhs) {
    return rhs * lhs;
}

// No *= for matrix * matrix multiplication, because it would be misleading: you cannot
// multiply a matrix in place, so we'd end up doing the same amount of work in regular
// multiplication and multiplication-assigment. Since there is an expectation that
// arithmetic assigments are done in place, we just don't provide the *= operator
template <size_t size>
matrix<size> operator*(const matrix<size>& lhs, const matrix<size>& rhs) {
    matrix<size> ret{};

    // Naive implementation, should probably use tiled multiplication for big matrices
    // (it could also be made multithreaded)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            auto& prod = ret(i, j);
            for (size_t k = 0; k < size; ++k) {
                prod += lhs(i, k) * rhs(k, j);
            }
        }
    }

    return ret;
}

template <size_t size>
matrix<size> operator/(const matrix<size>& lhs, double rhs) {
    matrix<size> ret{lhs};
    ret /= rhs;
    return ret;
}

// This probably shouldn't be used, == and != are terrible for float comparisons
template <size_t size>
bool operator==(const matrix<size>& lhs, const matrix<size>& rhs) {
    for (size_t row = 0; row < size; ++row) {
        for (size_t col = 0; col < size; ++col) {
            if (lhs(row,col) != rhs(row,col)) {
                return false;
            }
        }
    }
    return true;
}

template <size_t size>
bool operator!=(const matrix<size>& lhs, const matrix<size>& rhs) {
    return !(lhs == rhs);
}

// This way we can easily output matrices to a file or to cout
template <size_t size>
std::ostream& operator<<(std::ostream& out, const matrix<size>& m) {
    int pad_to = 10;

    out << std::fixed;
    for (size_t row = 0; row < m.size; ++row) {
        out << '|' << std::setw(pad_to) << m(row,0);
        for (size_t col = 1; col < m.size; ++col) {
            out << ", " << std::setw(pad_to) << m(row,col);
        }
        out << '|' << '\n';
    }
    return out << std::defaultfloat;
}
