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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


n1 = size(Theta1)(1,1);
n2 = size(Theta2)(1,1);

%diff = ones(m,1);

for i=1:m
		Xi = [1; X(i,:)'];
		A2 = Theta1*Xi;
		A2 = [1; sigmoid(A2)];
		A3 = Theta2*A2;
		A3 = sigmoid(A3);
		[dummy ,p] = max(A2, [],2);
		for k=1:num_labels
				if( y(i) == k) J = J - log(A3(k))/m;
				else J = J - log(1-A3(k))/m;
				endif
				%J = J - ( A2(k)*log(A2(k)) + (1-A2(k))*log(1-A2(k)))/m;
		end
end


for i=1:n1	%regularization Theta1
		temp1 = Theta1(i,:);
		temp1(1) = 0;
		J = J + lambda/2/m * ( temp1*temp1');
end

for i=1:n2	%regularization Theta2
		temp2 = Theta2(i,:);
		temp2(1) = 0;
		J = J + lambda/2/m * ( temp2*temp2');
end





% -------------------------------------------------------------

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.



D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));


		
for i=1:m
		Xi = [1; X(i,:)'];
		z2 = Theta1*Xi;
		a2 = [1; sigmoid(z2)];
		z3 = Theta2*a2;
		a3 = sigmoid(z3);
		yk = 1:num_labels;
		yk = yk == y(i);
		Delta3 = a3 - yk';
		Delta2 = (Theta2'*Delta3).*[0; sigmoidGradient(z2)];
		D2 = D2 + Delta3*a2';
		D1 = D1 + Delta2(2:end)*Xi';		
		
end
Theta1_grad = D1/m; 
Theta2_grad = D2/m;

J = J + sum(sum(Theta1)(2:end)) + sum(sum(Theta2)(2:end));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
