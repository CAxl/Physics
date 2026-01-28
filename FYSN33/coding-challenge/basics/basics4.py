# Basics 4: Count Words in a File
# Write a program to count the number of words in the file named basics4.txt.

def count_words_in_file(filename):
    count = 0
    with open(filename, "r") as f:
        data = f.read()
        words = data.split()
        count += len(words)
    return count

if __name__ == "__main__":
    word_count = count_words_in_file("basics4.txt")
    print(word_count)