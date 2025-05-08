import sys
import math
from collections import Counter

def read_data(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        data = []
        for line in file:
            parts = list(map(float, line.strip().split()))
            *features, label = parts
            data.append((features, int(label)))
        return data

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def get_majority_vote(neighbors):
    class_counts = Counter(label for _, label in neighbors)
    max_count = max(class_counts.values())
    tied_classes = [cls for cls, count in class_counts.items() if count == max_count]
    return tied_classes

def classify(training_data, test_data, k):
    results = []
    correct_count = 0
    accuracy_sum = 0.0

    for idx, (test_feat, true_label) in enumerate(test_data):
        # Calculate distances
        distances = [(euclidean_distance(test_feat, train_feat), label)
                     for train_feat, label in training_data]
        distances.sort(key=lambda x: x[0])
        nearest_k = distances[:k]

        # Voting
        tied_classes = get_majority_vote(nearest_k)

        # If tie, pick lowest label for deterministic output
        predicted_class = min(tied_classes)

        # Accuracy calculation
        if len(tied_classes) == 1:
            accuracy = 1.0 if predicted_class == true_label else 0.0
        else:
            accuracy = 1.0 / len(tied_classes) if true_label in tied_classes else 0.0

        results.append((idx, predicted_class, true_label, accuracy))
        accuracy_sum += accuracy

    avg_accuracy = accuracy_sum / len(test_data)
    return results, avg_accuracy

def main():
    if len(sys.argv) < 3:
        print("Usage: nn_classify <training-file> <test-file> [k]")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    training_data = read_data(training_file)
    test_data = read_data(test_file)

    results, avg_accuracy = classify(training_data, test_data, k)

    with open('result.txt', 'w') as f:
        for obj_id, pred_class, true_class, acc in results:
            f.write(f"objID = {obj_id} predicted class = {pred_class} true class = {true_class} accuracy = {acc:.5f}\n")

    print(f"Average accuracy: {avg_accuracy:.5f}")

if __name__ == "__main__":
    main()