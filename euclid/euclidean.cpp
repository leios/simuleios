/*-------------euclidean.cpp--------------------------------------------------//
*
* Purpose: To implement euclidean algorithm to find the greatest common divisor
*
*-----------------------------------------------------------------------------*/

#include <iostream>
#include <math.h>

// Euclidean algorithm with mod
int euclid_mod(int a, int b);

// Euclidean algorithm with subtraction
int euclid_sub(int a, int b);

// Euclidean algorith with recursion
int euclid_rec(int a, int b);

int main(){

    int check = euclid_mod(64*67, 64*81);
    int check2 = euclid_sub(128*12, 128*77);
    int check3 = euclid_rec(256*80, 256*11);

    std::cout << check << '\n';
    std::cout << check2 << '\n';
    std::cout << check3 << '\n';
}

// Euclidean algorithm with mod
int euclid_mod(int a, int b){

    int temp;
    while (b != 0){
        temp = b;
        b = a%b;
        a = temp;
    }

    return a;
}

// Euclidean algorithm with subtraction
int euclid_sub(int a, int b){

    while (a != b){
        if (a > b){
            a = a - b;
        }
        else{
            b = b - a;
        }
    }

    return a;
}

// Euclidean algorith with recursion
int euclid_rec(int a, int b){

    if (b == 0){
        return a;
    }
    else{
        return euclid_rec(b, a%b);
    }
}

