/* Basics 12: Read from a file in C++.
Get the number printed in the second line of report.txt, 
which you created in basics10.py. Take the exponential of
this number, and print it to screen. Make sure that your numerical 
precision matches the input file.
You can use the standard C++ libraries, such as cmath.
If the file is not found, the program should exit with code 1,
otherwise code 0.
*/

#include <iostream>
#include <fstream>
#include <cmath>

int main(int argc, char * argv[]){

    std::fstream file;
    std::string filename, line;
    float number;

    // TODO: Implements a program solving the exercise.

    return 0;
}
