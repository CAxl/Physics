/* Basics 15: Create a contact book
Write a program that manage a simple contact book.
The program can add a new contact or find a contact
by a name. The program runs continuously until
you ask to quit.
*/

#include <iostream>
#include <vector>
#include <string>

using std::string;
using std::vector;

// Structure to represent a contact with a name and a phone number.
struct contact
{
    string name;
    string phone_number;
};

// Vector to store the contacts in the directory.
vector <contact> directory;

// Function to display the menu and get the user's choice.
int menu(){
    // TODO: Implement this function
    return 0; // Replace with your implementation
}

// Function to add a new contact to the directory.
void add_contact(string name, string phone){
    // TODO: Implement this function
}

// Function to find a contact by name.
// Returns the contact if found or an empty contact if not found.
contact find_contact(string name){
    // TODO: Implement this function
    return; // Replace with your implementation
}

int main(){

    // TODO: Implements a program solving the exercise.

}
