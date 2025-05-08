Name: Chirag Naresh Patil
UTA ID: 1002223067

Programming Language Used:
Python 3.11.3 (Tested on Windows 11, should run on any system with Python 3.6+)

Project Overview:
This program implements Bayesian inference to estimate the posterior probabilities of five hypotheses regarding the composition of a bag of candies. Each hypothesis represents a different ratio of cherry and lime candies. The algorithm uses the Bayes' Theorem in a sequential manner — updating beliefs (posterior probabilities) after each new candy observation.

The task is also to compute the probability that the next candy picked from the bag will be of a specific type (cherry or lime), based on all prior observations and the posterior probabilities at each stage.

Code Structure:
- `compute_posterior.py`: This is the main script that:
    - Reads the command-line argument (observation sequence, e.g., CCLLL).
    - Initializes prior probabilities for five hypotheses (h1–h5).
    - Iteratively updates posterior probabilities after each observation.
    - At each step, calculates the probability that the next candy will be cherry or lime.
    - Prints the output in a formatted manner to `result.txt`, matching the instructor’s required format.
    

If you want to change the path to open folder for task 1 write cd "Task 1" and to go back to main folder write cd ..

Key Functions and Flow:
- `compute_posterior(observations)`: Implements Bayesian updates using recursive multiplication of likelihoods and normalizing over all hypotheses.
- `write_output(...)`: Writes structured results into `result.txt`, including before and after each observation with probabilities formatted to 5 decimal places.

How to Run:
To execute the program, use the following command:

    python compute_posterior.py CCLLLLCC

You can change the string of `C` and `L` characters to simulate any sequence of cherry or lime candies. Each character represents a candy drawn from the bag.

If no argument is passed:

    python compute_posterior.py

The script will default to displaying only the prior probabilities and initial prediction with no observations considered.

File Output:
- `result.txt`: This file will contain the complete formatted output including:
    - Priors before any observation
    - Posterior updates after each observation
    - Probability prediction for the next candy

Assumptions:
- The number of hypotheses and their priors are hardcoded, as per assignment instructions.
- Candy types are only cherry (C) or lime (L).
- The bag is assumed to be infinite, so replacement is implied.

Dependencies:
- Pure Python 3; no third-party packages are required.

Additional Notes:
- I ensured precise formatting in `result.txt` according to the sample PDF.
- The algorithm correctly handles floating-point normalization to prevent rounding errors.
- Posterior probabilities are normalized at each step to ensure they sum to 1.
