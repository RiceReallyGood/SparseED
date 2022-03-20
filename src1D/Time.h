#ifndef TIME_MANIPULATOR_H
#define TIME_MANIPULATOR_H

#include <iostream>
#include <iomanip>
#include <ctime>

std::ostream& Time(std::ostream& out){
    time_t t = std::time(nullptr);
    out << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    return out;
}

#endif //TIME_MANIPULATOR_H