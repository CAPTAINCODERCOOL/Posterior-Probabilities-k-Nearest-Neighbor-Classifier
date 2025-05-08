Name: Chirag Naresh Patil
UTA ID: 1002223067

Programming Language Used:
Python 3.11.3 (Tested on Windows 11; compatible with any Python 3+ environment)

Project Overview:
This program implements a K-Nearest Neighbor (KNN) classification algorithm. The goal is to classify objects in a test dataset by comparing each one to a set of labeled training examples. The algorithm calculates the Euclidean distance between the test object and each training object, identifies the closest k neighbors, and assigns the most common class among them to the test object.

Additionally, the program evaluates accuracy on a per-object basis with detailed handling of ties, as per the assignment requirements, and prints a summary of overall accuracy.

Code Structure:
- `nn_classify.py`: This is the main script that:
    - Parses command-line arguments: training file, test file, and optional `k` (number of neighbors, default = 1).
    - Reads the training and testing data files, both in whitespace-separated format.
    - For each test object:
        - Computes Euclidean distance to every training object.
        - Finds the k closest objects and performs majority voting.
        - Handles ties by selecting the lowest label from the tied classes (or random selection if required).
        - Compares the predicted label with the actual label and assigns accuracy according to strict rules.
    - Writes each result line to `result.txt` and prints overall accuracy to the console.

Key Functions:
- `read_data()`: Parses and loads the training/test files into a list of (features, label) tuples.
- `euclidean_distance()`: Computes distance between two points in feature space.
- `classify()`: Core KNN logic, includes tie-breaking and per-object accuracy computation.
- `main()`: Handles execution flow and file writing.

If you want to change the path to open folder for task 2 write cd "Task 2" and to go back to main folder write cd ..

How to Run:
Run the classifier with this command:

    python nn_classify.py pendigits_training.txt pendigits_test.txt 3

Where:
- `pendigits_training.txt`: File containing labeled training data.
- `pendigits_test.txt`: File containing test data to classify.
- `3`: (Optional) Number of neighbors to use (k=3). If omitted, defaults to 1.

File Output:
- `result.txt`: A line-by-line output for each test object, formatted exactly like the sample:
    - `objID = <index>` (starting from 0)
    - `predicted class = <value>`
    - `true class = <value>`
    - `accuracy = <score>` (can be 0.0, 1.0, or fractional if tied)
- Prints **overall average accuracy** to standard output (terminal).

Example Output Line:
    objID = 27 predicted class = 4 true class = 4 accuracy = 1.0

Tie-Handling:
- If there's a tie in the top k class votes:
    - If the true label is one of the tied labels → accuracy = 1 / (number of tied classes)
    - If the true label is not among the tied → accuracy = 0.0
    - If no tie and prediction is correct → accuracy = 1.0
    - If no tie and incorrect → accuracy = 0.0

Assumptions:
- The training and test files are in plain text, with the class label in the last column.
- All features are numerical and whitespace-separated.

Dependencies:
- Pure Python standard libraries; no external packages are required.

Additional Notes:
- I tested the program using the `pendigits` dataset provided in the assignment.
- The code is structured for readability and simplicity, with separate sections for data loading, classification, and evaluation.
- I have thoroughly validated the format of the output to match the instructor's example.
