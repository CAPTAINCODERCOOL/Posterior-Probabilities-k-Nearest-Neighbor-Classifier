Posterior Probabilities & k-Nearest Neighbor Classifier

This repository contains implementations for two machine learning tasks. You may complete either one for full credit or both for extra credit.

---

## 🧠 Task 1: Posterior Probability Inference

This task implements a Bayesian inference system that:
- Computes posterior probabilities for five hypotheses (types of candy bags) based on a given observation sequence.
- Computes the probability that the next candy is cherry or lime.

### 🏷 Hypotheses:
| Hypothesis | Description                          | Prior  |
|------------|--------------------------------------|--------|
| h1         | 100% cherry candies                  | 0.10   |
| h2         | 75% cherry, 25% lime                 | 0.20   |
| h3         | 50% cherry, 50% lime                 | 0.40   |
| h4         | 25% cherry, 75% lime                 | 0.20   |
| h5         | 100% lime candies                    | 0.10   |

### ✅ How to Run:
```bash
python compute_posterior.py CCLLLLLLCCCC
(If no argument is given, assumes zero observations.)

🧾 Output:
Generates result.txt with:

Posterior probabilities after each observation

Prediction probabilities for next candy (C or L)

🤖 Task 2: k-Nearest Neighbor Classifier
This task implements a k-NN classifier from scratch. Supports 1-NN by default or custom k value.

📂 Datasets:
Pendigits (16 features, 10 classes)

Satellite (36 features, 6 classes)

Yeast (8 features, 10 classes)

✅ How to Run:
bash
Copy
Edit
python nn_classify.py <training-file> <test-file> [<k>]
Example:

bash
Copy
Edit
python nn_classify.py datasets/pendigits_training.txt datasets/pendigits_test.txt 3
🧾 Output:
Creates a result.txt file listing:

Object ID

Predicted class

True class

Accuracy (1, 0, or fractional if tie)

Also prints overall classification accuracy to console.

📁 File Structure
bash
Copy
Edit
.
├── compute_posterior.py           # Task 1 script
├── nn_classify.py                 # Task 2 script
├── datasets/                      # Folder containing dataset files
├── result.txt                     # Output file for both tasks
├── LICENSE                        # Apache 2.0 License
└── README.md                      # This file
🔓 License
This project is licensed under the Apache License 2.0 – see LICENSE for details.
