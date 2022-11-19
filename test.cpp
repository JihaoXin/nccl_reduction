#include<iostream>
using namespace std;
int main(void)
{
    int a = 12;
    cout << (1==2)<<endl;
    cout << (1==1)<<endl;
    cout << a - (1==2)<<endl;
    cout << a - (1==1)<<endl;
}