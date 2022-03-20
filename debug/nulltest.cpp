#include <cstring>
#include <iostream>

int main() {
    int N = 10;
    int** nptr = new int*[N];
    memset(nptr, 0, sizeof(int*));

    std::cout.setf(std::ios::boolalpha);
    for (int i = 0; i < N; i++) {
        std::cout << (nptr[i] == nullptr) << std::endl;
    }
}