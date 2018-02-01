function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
JReg = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


for i = 1:m

  Xin = X(i,:);
  Xin = [1, Xin];
    
  yVals = [1:num_labels]';
  yVals = (yVals == y(i));
  
  z2 = Theta1*Xin';
  z2 = [1; z2];
  hidden = sigmoid(z2);
  
  z3 = Theta2*hidden;
  output = sigmoid(z3);
  
  d3 = output - yVals;
  
  d2 = (Theta2')*d3;
  
  d2 = d2.*sigmoidGradient(z2);
  
  J = J + log(output)'*yVals + log(1 - output)'*(1 - yVals);
  
  Theta1_grad = d2*Xin;
  Theta2_grad = d3*(z2');
  
end;

  J = -J/m;
  
  RegTheta1 = Theta1(:,2:size(Theta1, 2));
  RegTheta2 = Theta2(:,2:size(Theta2, 2));
  
  JReg = JReg + sum(sum(RegTheta1.^2)) + sum(sum(RegTheta2.^2));
  JReg = JReg/(2*m);
  
  J = J + JReg;
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
