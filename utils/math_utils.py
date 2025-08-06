import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def find_optimal_threshold(y_true, y_pred_proba, num_thresholds=100, metric='f1'):
    """
    Finds the optimal threshold for binary classification based on maximizing
    either F1-score or accuracy.

    Args:
        y_true (np.array): Array of true binary labels (0 or 1).
        y_pred_proba (np.array): Array of predicted probabilities (values between 0 and 1).
        num_thresholds (int): The number of thresholds to test between 0 and 1.
        metric (str): The metric to optimize ('f1' or 'accuracy'). Default is 'f1'.

    Returns:
        tuple: A tuple containing:
            - best_threshold (float): The threshold that maximizes the chosen metric.
            - best_score (float): The maximum score achieved for the chosen metric.
            - corresponding_metric_value (float): Accuracy score if optimizing for F1,
                                                   or F1 score if optimizing for accuracy.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    # Define thresholds to check
    # Using unique prediction probabilities + epsilon might be more efficient
    # thresholds = np.sort(np.unique(y_pred_proba))
    # thresholds = np.append(thresholds, thresholds[-1] + 1e-9) # Add a value slightly higher
    # thresholds = np.insert(thresholds, 0, thresholds[0] - 1e-9) # Add a value slightly lower
    # However, linspace is simpler to understand and often sufficient
    thresholds = np.linspace(0, 1, num_thresholds + 1) # Includes 0 and 1

    best_threshold = 0
    best_score = -1
    corresponding_accuracy = -1
    corresponding_f1 = -1

    # Iterate through thresholds
    for threshold in thresholds:
        # Apply threshold to get binary predictions
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred_binary)
        # Handle cases with no positive predictions/labels for F1
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        # Determine the primary score to optimize
        current_primary_score = f1 if metric == 'f1' else acc

        # Check if this threshold gives a better primary score
        if current_primary_score > best_score:
            best_score = current_primary_score
            best_threshold = threshold
            corresponding_accuracy = acc
            corresponding_f1 = f1
        # Tie-breaking rule (optional): If primary scores are equal,
        # prefer the one with the better secondary score.
        elif current_primary_score == best_score:
            if metric == 'f1' and acc > corresponding_accuracy:
                 # If optimizing F1 and F1s are tied, choose higher accuracy
                 best_threshold = threshold
                 corresponding_accuracy = acc
                 corresponding_f1 = f1 # f1 is already equal to best_score
            elif metric == 'accuracy' and f1 > corresponding_f1:
                 # If optimizing Accuracy and Accuracies are tied, choose higher F1
                 best_threshold = threshold
                 corresponding_accuracy = acc # acc is already equal to best_score
                 corresponding_f1 = f1


    print(f"Optimization based on: {metric.upper()}")
    if metric == 'f1':
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Best F1-Score: {best_score:.4f}")
        print(f"Corresponding Accuracy: {corresponding_accuracy:.4f}")
        return best_threshold, best_score, corresponding_accuracy
    else: # metric == 'accuracy'
        print(f"Best Threshold: {best_threshold:.4f}")
        print(f"Best Accuracy: {best_score:.4f}")
        print(f"Corresponding F1-Score: {corresponding_f1:.4f}")
        return best_threshold, best_score, corresponding_f1


