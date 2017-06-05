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


n =  size(theta,1);



for index1=1:m  %j
		gx = X(index1,:)*theta;
		J = J - ( y(index1)*log(sigmoid(gx)) + (1-y(index1))*log(1-sigmoid(gx)))/m;		
		for i=1:n
				grad(i) =  grad(i) + ( sigmoid(gx)- y(index1) )* X(index1,i)/m ; 
				
		end
		
end


for i=2:n  
			grad(i) =  grad(i) +  lambda/m*theta(i,1); 	
			J = J + lambda * theta(i,1)^2 /2/m;				
end



% =============================================================

end
