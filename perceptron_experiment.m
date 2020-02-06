function [ num_iters, bounds] = perceptron_experiment ( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bounds is the theoretical bound on the # of iterations
%              for each sample
%      (both the outputs should be num_samples long)
num_iters = zeros(num_samples,1);
bounds = zeros(num_samples,1);
for i = 1:num_samples
    w = rand(d+1,1);
    w(1,1) = 0;
    x = ones(N,d+1)-rand(N,d+1)*2;
    x(:,1) = ones(N,1);
    y = sign(x*w);
    bounds(i,1) = max(max(x*x.'))*(w.'*w)/min((x*w).^2);
    z = [x,y];
    [w_train, iterations] = perceptron_learn(z);
    num_iters(i,1)=iterations;
end
end
function [ w, iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
    w_train = zeros(size(data_in,2)-1,1);
    x = data_in(:,1:(size(data_in,2)-1));
    y = data_in(:,size(data_in,2));
    error = sign(x*w_train)-y;
    ite = 0;
    while error.'*error ~= 0
        position = find(error,1);
        w_train = w_train + sign(y(position,1))*x(position,:).';
        ite = ite+1;
        error = sign(x*w_train)-y;
    end
    w = w_train;
    iterations = ite;
end
