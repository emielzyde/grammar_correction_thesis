import numpy as np

def precision(predictions, targets):
    'Expects input and output numpy arrays which contain only 0s and 1s'

    #The product of two binary (0/1) numbers is only 1 if they are both 1.
    #Thus, the number of 1s predicted correctly is the sum of the product of the arrays;
    true_positive = np.sum(predictions * targets)
    total_predicted_positive = np.sum(predictions)

    precision = true_positive/total_predicted_positive
    return precision

def recall(predictions, targets):
    true_positive = np.sum(predictions * targets)
    total_actual_positive = np.sum(targets)

    recall = true_positive/total_actual_positive
    return recall

def f_score(precision, recall, beta):
    numerator = (1 + beta**2) * precision * recall
    denominator = beta**2 * precision + recall

    f_score = numerator/denominator
    return f_score

def calculate_f_score(predictions, targets, beta = 0.5):
    prec = precision(predictions, targets)
    rec = recall(predictions, targets)
    f_scorer = f_score(prec, rec, beta)

    return f_scorer

if __name__ == '__main__':
    a = np.array([0,1,1,0,1,0,0,0])
    b = np.array([0,1,0,1,0,0,0,0])
    print('Precision', precision(a,b))
    print('Recall', recall(a,b))
    prec = precision(a,b)
    rec = recall(a,b)
    print('F_0.5', f_score(prec,rec,0.5))
