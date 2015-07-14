/*------------hello_gpu.cu----------------------------------------------------//
*
*             hello_gpu
*
* Purpose: Hello world with our first kernel on the side!
*
*-----------------------------------------------------------------------------*/

#include <iostream>

using namespace std;

__global__ void kernel(void){
}

int main(void){

    kernel<<<1,1>>>();
    cout << "hey guys. I thought we were supposed to be printing off the gpu..."
         << "but I was wrong. =(" << endl;
    return 0;
}    
