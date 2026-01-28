# Basics 5: Leap Year
# Write a program that prints out a list of all leap years from the year 1850 to 2050

# The definition of a leap year in the Gregorian Calendar is (from Wikipedia)
# Every year that is exactly divisible by four is a leap year, except for years that are exactly 
# divisible by 100, but these centurial years are leap years if they are exactly divisible by 400. 
# For example, the years 1700, 1800, and 1900 are not leap years, but the years 1600 and 2000 are.

if __name__ == "__main__":
    leap_years = range(1850,2051)
    # TODO: Implement a body which filters away non-leap years.
    for l in leap_years:
    	print(l)
