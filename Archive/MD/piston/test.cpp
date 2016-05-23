#include <iostream>

using namespace std;

void f(int& x){
    x++;
}

int main(){

    int x = 4;
    f(x);
    cout << x << endl;
}
