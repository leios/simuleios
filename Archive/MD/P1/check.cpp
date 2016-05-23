/*-------------check.cpp------------------------------------------------------//
*
*              check.cpp
*
* Purpose: To answer a simple question: What is more efficient? New vs old
*
*-----------------------------------------------------------------------------*/

#include<iostream>
#include<ctime>
#include<vector>
#include<random>
#include<algorithm>

using namespace std;

int main(void){

    clock_t start;

    // first checking loops
    int n = 1000000, sum;
    double time1, time2, time3;
    vector <double> array(n, false);

    time1 = (clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
    cout << time1 << endl;

    sum = 0;

    for (int i = 0; i < n; i++){
        sum++;
        sum = sum * (sum / sum);
        array[i] = sum;
    }

    time2 = (clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
    cout << "check 1: " << time2 - time1 << endl;

    sum = 0;

    for (auto &p : array){
        sum++;
        sum = sum * (sum / sum);
        p = sum;
    }

    time3 = (clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
    cout << "check 1: " << time3 - time1 - time2 << endl;

}
