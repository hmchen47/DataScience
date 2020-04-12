"""
URL: https://trongr.github.io/neural-network-course/neuralnetworks.html
"""

learning_rates = [3.5e-7, 2e-7, 1.7e-7, 1.5e-7]
regularization_strengths = [5e3, 8e3, 1e4, 2e4]

results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.

def validateHyperparams(X_train, y_train, X_val, y_val, alpha, reg):
    svm = LinearSVM()
    svm.train(X_train, y_train, learning_rate=alpha, reg=reg, num_iters=1500) # todo change num_iters to 1500
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)
    train_acc = np.mean(y_train == y_train_pred)
    val_acc = np.mean(y_val == y_val_pred)
    return train_acc, val_acc, svm

for alpha in learning_rates:
    for reg in regularization_strengths:
        train_acc, val_acc, svm = validateHyperparams(X_train, y_train, X_val, y_val, alpha, reg)
        results[(alpha, reg)] = (train_acc, val_acc)
        if (val_acc > best_val):
            best_val = val_acc
            best_svm = svm
