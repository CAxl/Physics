# Basics 2: FizzBuzz
# Write a program that prints numbers 1-20 with FizzBuzz rules:
# - Multiples of 3: print "Fizz"
# - Multiples of 5: print "Buzz"
# - Multiples of both 3 and 5: print "FizzBuzz"

def fizzbuzz():
	for i in range(1,21):
		if i%3 == 0 and i%5 == 0:
			print("FizzBuzz")
		elif i%3 == 0:
			print("Fizz")
		elif i%5 == 0:
			print("Buzz")
		else:
			print(i)

if __name__ == "__main__":
	fizzbuzz()
