# Coding challenge


## The challenge!

This repo consists of a series of coding challenges in Python and C++. They are intended to all be solvable with max. 50-100 lines of code, and without use of external libraries, unless explicitly stated in the challenge.

The coding challenges are self-correcting. A CI script will run your solution, and give you immidiate feedback on whether your solution is correct or not.

To get started, read on below.

## Starting the challenge

- [ ] [Familiarize yourself](https://docs.gitlab.com/ee/topics/git/) with git and gitlab. It is a point of this challenge that not only do you get to brush up on your coding skills, you will also get practical experience using gitlab.
- [ ] Make sure you are sitting on a machine with Python and a C++ compiler installed, access to a terminal and have git installed.
- [ ] Make sure that you have [added an ssh key](https://docs.gitlab.com/ee/user/ssh.html) to communicate with Gitlab.
- [ ] Make a [feature branch](https://docs.gitlab.com/ee/user/project/repository/branches/) off the existing branch called [student-solution-template](https://gitlab.com/comp-phys-ai/coding-challenge/-/tree/student-solution-template). This will hold template code and challenge descriptions for all challenges. Use the following commands in a terminal:
```
git clone --branch student-solution-template git@gitlab.com:comp-phys-ai/coding-challenge.git
cd coding-challenge/
git checkout -b my-awesome-solutions # You can use a branch name of your choice instead of this
```

You can then begin coding, i.e. solving the exercises. Some of the exercises will have detailed instructions in the readme, but most of them will be self-explanatory from the code and the comments in the code. When you have solved a couple of exercises to your satisfaction, make a merge request to the master branch. You do this by committing your changes like this.

```
# Say you have solved basics1.py. Do this:
git add basics1.py
git commit -m "committing solution of basics1.py"
git push
```
Your terminal gives you an error message here, first time you push. Read it. Follow it:
```
git push --set-upstream origin my-awesome-solutions
```
Your terminal now tells you what to do:
```
remote: To create a merge request for my-awesome-solutions, visit:
remote:   https://gitlab.com/comp-phys-ai/coding-challenge/-/merge_requests/new?merge_request%5Bsource_branch%5D=my-awesome-solutions
```
So do it. Then go to gitlab and inspect your branch.

## Checking results
Gitlab runs control scripts to check the output of your code. Use it to validate your solutions.

Does this not mean that you can just glean the correct solutions from the tests? Yes it does. You could also just ask ChatGPT, your friend, StackOverflow etc. Don't.

## Go on and solve
You should then go on and solve all exercises and commit them to your merge request. Once you are done, you are invited to come up with a challenge on your own and commit it, along with a description of features added to your merge request. We will include (variations of...) the best suggestions for next time.

## Authors and contributions
These coding exercises are written by Christian Bierlich (christian.bierlich@fysik.lu.se) and Lise Auffret. Contributions are welcome!

## License
MIT License

Copyright (c) 2024 Christian Bierlich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.