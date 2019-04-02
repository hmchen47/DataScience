% Create & Call function with individual file
function y = squareThisNumber(x)

y = x^2;


% Navigate to directory:
cd /path/to/function

% Call the function:
functionName(args)


% To add the path for the current session of Octave:
addpath('/path/to/function/')

% To remember the path for future sessions of Octave, after executing addpath above, also do:
savepath


% individual file
function [y1, y2] = squareandCubeThisNo(x)
y1 = x^2
y2 = x^3

[a,b] = squareandCubeThisNo(x)

