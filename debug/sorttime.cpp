#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Please input data size N!" << std::endl;
        std::cout << "Usage: ./sorttime N" << std::endl;
        exit(1);
    }

    int N = std::stoi(argv[1]);

    int* data = new int[N];
    std::default_random_engine dre;
    std::uniform_int_distribution<int> uid(0, N - 1);
    for (int i = 0; i < N; i++) data[i] = uid(dre);
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::sort(data, data + N);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Elapsed time: " << diff.count() << "s" << std::endl;

    delete[] data;
}