function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

z = X*theta;
h = sigmoid(z);
J = (-1/m)*((y')*(log(h)) + (1 - y')*(log(1 - h)));
thetaReg = theta;
thetaReg(1) = 0;

JReg = (lambda/m)*(1/2)*(thetaReg'*thetaReg);
J = J + JReg;

gradReg = (lambda/m)*thetaReg;

grad = (1/m)*X'*(h - y);
grad = grad + gradReg;

end
