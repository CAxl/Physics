# Basics

The challenges in this group are designed around using basic language features in Python. 

## Instructions
1. Make a [fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html) of this project, if you have not done so already (following the directions on the front page).
2. Edit the file(s) `basics[1-10].py` and push the changes to your forked project.
3. Check the CI pipeline to confirm that you have solved the challenges correctly.
4. Once you are satisfied, make an upstream merge request.

Below follows instructions for the basic challenges.

## Basics 1 (`basics1.py`): Print Numbers 1-10

## Task
Write a Python program that prints the numbers 1 to 10, each on a new line.

## Requirements
- Use a `for` loop to iterate through the numbers.
- Do not hard-code the output (e.g., avoid manually writing `print(1)`, `print(2)`, etc.).

## Hints
This task is meant for you to familiarize yourself with the environment. There are no hints.

## Example Output
The program should produce the following output:
```
1
2
3 
4 
5 
6 
7 
8 
9 
10
```

## Basics 8 (`basics8.py`): Calculate factorial of 1,2,3,4

## Task
Write a Python program that calculate the factorial of a number. Print the 
factorial for 1, 2, 3, 4.

## Requirements
- Use `for` loops

## Example Output
The program should produce the following output:
```
1
2
6
24
```

## Basics 9 (`basics9.py`): Program an encoder

## Task
Write a Python program that change a word (input argument).The program should change all instances
of the letter 'a' to '@' and the letter 't' to '!'.

## Requirements
- Use the `sys` module
- Check if there is an input argument


## Example Output
If the input argument is "airport", the program should produce the following output:
```
@irpor!
```

## Basics 10 (`basics10.py`): Program an encoder

## Task
Write a Python program that create a file called 'report.txt'. The contents of the file should be:
New report
the natural logarithm of the number 5


## Requirements
- Use the `numpy` library
- Use `artifacts` functionality in the CI test
- For the CI test: check the case if the file is created and its contents


## Basics 11 (`basics11.cpp`): Count the number of vowels in a word.

## Task
Write a C++ program that count the vowels of a word. The word is an input argument.


## Requirements
- Use the `string` library
- Check if there is an input argument

## Example Output
If the input argument is "hello", the program should produce the following output:
```
2
```

## Basics 12 (`basics12.cpp`): Read from a file in C++.

## Task
Write a C++ program that read the second line of report.txt (file created in basics10.py).
Print the exponential of this number. The filename "report.txt" should be an input argument.

## Requirements
- Use the `cmath` and `fstream` library
- Check if there is an input argument
- Check if the file exists
- Use `dependencies` for the CI test

## Example Output
The program should produce the following output:
```
5
```

## Basics 13 (`basics13.cpp`): Find the prime numbers.

## Task
Write a C++ program that find the prime numbers from 0 to 20.
Print them, each in a new line.

## Requirements
- Use the `fstream` library


## Example Output
The program should produce the following output:
```
2
3
5
7
11
13
17
19

```

## Basics 14 (`basics14.py`): Find the prime numbers.

## Task
Write a python program that retrieves data from the website : https://api.github.com.
Print the status code of the request.

## Requirements
- Use the `request` library


## Example Output
The program should produce the following output:
```
200

```

## Basics 15 (`basics15.cpp`): Create a contact book.

## Task
Write a C++ program that manages a simple contact book. The user can add new contacts or search for
existing ones by name. The program should continue running until the user chooses to quit.

## Requirements
- Display a menu to the user with three choices:
    1 - Add a contact
    2 - Find a contact by name
    3 - Quit the program
- Use a loop to repeatedly display the menu and perform actions based on the user’s input.

## Hints
- You can return an empty contact to indicate that a contact was not found.


## Example Output
The program should produce the following output:
```
1 - Add a contact  
2 - Find a contact  
3 - Quit  
Your choice : 1  
Enter the name : Alice  
Enter the phone number : 123456789  
Contact add  

1 - Add a contact  
2 - Find a contact  
3 - Quit  
Your choice : 2  
Enter the name : Alice  
Name : Alice  
Phone : 123456789  

1 - Add a contact  
2 - Find a contact  
3 - Quit  
Your choice : 3  
End  

```