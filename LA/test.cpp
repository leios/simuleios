#define CATCH_CONFIG_MAIN

// This is the unit testing framework we're gonna use to check if our matrix
// implementation is correct. This is a 3rd party library.
#include "catch.hpp"
#include "matrix.h"

// These test cases aren't meant to be industrial strength. We should probably
// randomly generate test cases and specify ones for edge cases. It's mainly here
// to show the basic idea

TEST_CASE("Matrix initialization and indexing", "[matrix]") {
    matrix<3> m = {0, 1, 2,
                   3, 4, 5,
                   6, 7, 8};

    REQUIRE(m(0,0) == 0.0);
    REQUIRE(m(0,1) == 1.0);
    REQUIRE(m(0,2) == 2.0);

    REQUIRE(m(1,0) == 3.0);
    REQUIRE(m(1,1) == 4.0);
    REQUIRE(m(1,2) == 5.0);

    REQUIRE(m(2,0) == 6.0);
    REQUIRE(m(2,1) == 7.0);
    REQUIRE(m(2,2) == 8.0);

    m(0, 0) = 10.0;
    REQUIRE(m(0,0) == 10.0);
}

TEST_CASE("Arithmetic assigments", "[matrix]") {
    matrix<3> m = {0, 1, 2,
                   3, 4, 5,
                   6, 7, 8};

    SECTION("matrix += matrix") {
        auto actual = m;
        actual += m;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == m(i,j) + m(i,j));
            }
        }
    }

    SECTION("matrix -= matrix") {
        auto actual = m;
        actual -= m;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == 0.0);
            }
        }
    }

    SECTION("matrix *= scalar") {
        auto actual = m;
        double s = 2.0;
        actual *= s;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == m(i,j) * s);
            }
        }
    }

    SECTION("matrix /= scalar") {
        auto actual = m;
        double s = 2.0;
        actual /= s;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == Approx(m(i,j) / s));
            }
        }
    }
}

TEST_CASE("Identity", "[matrix]") {
    auto m = identity<3>();
    for (size_t i = 0; i < m.size; ++i) {
        for (size_t j = 0; j < m.size; ++j) {
            if (i == j) {
                REQUIRE(m(i,j) == 1.0);
            } else {
                REQUIRE(m(i,j) == 0.0);
            }
        }
    }
}

TEST_CASE("Determinant", "[matrix]") {
    SECTION("2x2") {
        matrix<2> m = {1, 2,
                       3, 4};
        auto determinant = det(m);
        REQUIRE(determinant == 1.0 * 4.0 - 2.0 * 3.0);
    }

    SECTION("3x3") {
        matrix<3> m = {1, 2, 3,
                       7, 5, 6,
                       4, 8, 9};
        auto determinant = det(m);
        REQUIRE(determinant == Approx(27.0));
    }
}

TEST_CASE("Trace", "[matrix]") {
    auto m = identity<10>();
    REQUIRE(trace(m) == 10.0);
}

TEST_CASE("Comparison operators", "[matrix]") {
    matrix<3> m = {0, 1, 2,
                   3, 4, 5,
                   6, 7, 8};
    matrix<3> m2 = {0, 1, 2,
                    3, 5, 5,
                    6, 7, 8};
    REQUIRE(m == m);
    REQUIRE(m != m2);
}

TEST_CASE("Arithmetic operators", "[matrix]") {
    matrix<3> m = {0, 1, 2,
                   3, 4, 5,
                   6, 7, 8};

    SECTION("matrix + matrix") {
        auto actual = m + m;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == m(i,j) + m(i,j));
            }
        }
    }

    SECTION("matrix -= matrix") {
        auto actual = m - m;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == 0.0);
            }
        }
    }

    SECTION("matrix *= scalar") {
        auto actual = m * 2.0;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == m(i,j) * 2.0);
            }
        }
    }

    SECTION("scalar * matrix") {
        auto actual = 2.0 * m;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == m(i,j) * 2.0);
            }
        }
    }

    SECTION("matrix * matrix") {
        auto actual = m * m;
        auto i = identity<3>();
        matrix<3> expected = {15, 18, 21,
                              42, 54, 66,
                              69, 90, 111};
        REQUIRE(actual == expected);
        REQUIRE(m * i == i * m);
        REQUIRE(m * i == m);
    }

    SECTION("matrix /= scalar") {
        auto actual = m / 2.0;
        for (size_t i = 0; i < actual.size; ++i) {
            for (size_t j = 0; j < actual.size; ++j) {
                REQUIRE(actual(i,j) == Approx(m(i,j) / 2.0));
            }
        }
    }
}
