#include <iostream>

using namespace std;

class A {
public:
    virtual void fun();
};

int main() {
    cout << sizeof(A) << endl;
    return 0;
}