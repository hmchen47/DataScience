function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
C_set = [0.01 0.03 0.1 0.3 1 3 10 30];
% sigma = 0.3;
s_set = [0.01 0.03 0.1 0.3 1 3 10 30];

results = zeros(size(C_set, 2) * size(s_set, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for c_iter = 1:size(C_set, 2)
    for s_iter = 1:size(s_set, 2)
        
        model = svmTrain(X, y, C_set(c_iter), ...
            @(x1, x2) gaussianKernel(x1, x2, s_set(s_iter)));
        
        predictions = svmPredict(model, Xval);
        
        results((c_iter - 1) * size(C_set, 2) + s_iter) = mean(double(predictions ~= yval));
    end;
end;

[rlt, idx] = min(results);

C = C_set(idivide(idx, size(C_set, 2)));

sigma = s_set(mod(idx, size(C_set, 2)));

% =========================================================================

end
