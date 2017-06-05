function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



n =  size(theta,1);

for i=1:m
		hx = X(i,:)*theta;
		J = J + 1/2/m*(hx-y(i))^2;
		for j=1:n
				grad(j) =  grad(j) + (hx-y(i))* X(i,j)/m ; 
		end
end


for i=2:n  
			grad(i) =  grad(i) +  lambda/m*theta(i,1); 	
			J = J + lambda * theta(i,1)^2 /2/m;				
end












% =========================================================================

grad = grad(:);

end
