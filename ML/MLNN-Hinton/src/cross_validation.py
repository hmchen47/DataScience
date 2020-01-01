URL: https://trongr.github.io/neural-network-course/neuralnetworks.html

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

def trainAndValidate(X_train, y_train, X_val, y_val, k):
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances_no_loops(X_val)
    y_val_pred = classifier.predict_labels(dists, k)
    num_correct = np.sum(y_val_pred == y_val)
    num_val = len(y_val)
    return float(num_correct) / num_val

for k in k_choices:
    k_to_accuracies[k] = []
    for i in xrange(num_folds):
        Xtr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:], axis=0)
        ytr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:], axis=0)
        Xval = X_train_folds[i]
        yval = y_train_folds[i]
        k_to_accuracies[k].append(trainAndValidate(Xtr, ytr, Xval, yval, k))

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)

