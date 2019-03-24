function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[m, n] = size(X)

J = 1.0 / m * (-1.0 * y' * log(sigmoid(X * theta)) - ...
    (1.0 - y)' * log(1.0 - sigmoid(X * theta))) + ...
    lambda * (theta' * theta - theta(1)^2) / (2.0 * m);
    
for iter=1:n,
    if iter == 1,
        grad(iter) = X(:, iter)' * (sigmoid(X * theta) - y) / m;
    else
        grad(iter) = X(:, iter)' * (sigmoid(X * theta) - y) / m + ...
            lambda * theta(iter) / m;
    end;
end;

% =============================================================

end
