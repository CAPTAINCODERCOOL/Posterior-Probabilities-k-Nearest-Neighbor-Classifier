import sys

# Define hypotheses
hypotheses = ['h1', 'h2', 'h3', 'h4', 'h5']

# Priors
priors = {
    'h1': 0.10,
    'h2': 0.20,
    'h3': 0.40,
    'h4': 0.20,
    'h5': 0.10
}

# Likelihoods
likelihoods = {
    'h1': {'C': 1.0, 'L': 0.0},
    'h2': {'C': 0.75, 'L': 0.25},
    'h3': {'C': 0.50, 'L': 0.50},
    'h4': {'C': 0.25, 'L': 0.75},
    'h5': {'C': 0.0, 'L': 1.0}
}

def normalize(probs):
    total = sum(probs.values())
    for h in probs:
        probs[h] /= total
    return probs

def compute_posterior(observations):
    posteriors = [priors.copy()]
    predictions = []

    for t, obs in enumerate(observations):
        prev = posteriors[-1]
        numerators = {}
        for h in hypotheses:
            numerators[h] = likelihoods[h][obs] * prev[h]
        
        normalizer = sum(numerators.values())
        new_post = {h: numerators[h] / normalizer for h in hypotheses}
        posteriors.append(new_post)

        # Predict next candy probabilities
        p_c = sum(likelihoods[h]['C'] * new_post[h] for h in hypotheses)
        p_l = sum(likelihoods[h]['L'] * new_post[h] for h in hypotheses)
        predictions.append((p_c, p_l))

    return posteriors, predictions

def write_output(observations, posteriors, predictions):
    with open("result.txt", "w") as f:
        f.write(f"Observation sequence Q: {observations}\n")
        f.write(f"Length of Q: {len(observations)}\n\n")

        f.write("Before Observations:\n")
        for h in hypotheses:
            f.write(f"P({h}) = {priors[h]:.5f}\n")

        p_c = sum(likelihoods[h]['C'] * priors[h] for h in hypotheses)
        p_l = sum(likelihoods[h]['L'] * priors[h] for h in hypotheses)
        f.write(f"\nProbability that the next candy we pick will be C, given Q: {p_c:.5f}\n")
        f.write(f"Probability that the next candy we pick will be L, given Q: {p_l:.5f}\n\n")

        for t, obs in enumerate(observations):
            f.write(f"After Observation {t+1} = {obs}:\n")
            for h in hypotheses:
                f.write(f"P({h} | Q) = {posteriors[t+1][h]:.5f}\n")
            f.write(f"\nProbability that the next candy we pick will be C, given Q: {predictions[t][0]:.5f}\n")
            f.write(f"Probability that the next candy we pick will be L, given Q: {predictions[t][1]:.5f}\n\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        observations = sys.argv[1].strip().upper()
        valid = all(c in ['C', 'L'] for c in observations)
        if not valid:
            print("Error: Observation sequence must contain only 'C' or 'L'.")
            sys.exit(1)
    else:
        observations = ""

    posteriors, predictions = compute_posterior(observations)
    write_output(observations, posteriors, predictions)