# Basics 3: Average of a list
# Write a program that calculates the average of a list of numbers.

def average_list(numbers):
    s = 0.0
    for val in numbers:
        s += val
    avg = s/len(numbers)
    return avg

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5, 6]
    print(average_list(numbers))