function a3(wd_coefficient, n_hid, n_iters,learning_rate,momentum_multiplier, do_early_stopping, mini_batch_size)
  warning('error', 'Octave:broadcast');
  disp('Inside the a3 function')
  if exist('page_output_immediately'), page_output_immediately(1); end
  more off;
  model = initial_model(n_hid);  
  disp('size(model.input_to_hid): '), disp(size(model.input_to_hid))
  disp('size(model.hid_to_class): '), disp(size(model.hid_to_class))
  % Will contain the weights from input to hidden and hidden to output unit
  
  from_data_file = load('data.mat');
  datas = from_data_file.data;
  n_training_cases = size(datas.training.inputs, 2);  % 1000-> number of training cases
  % datas.training.inputs = 256 x 1000 (thousand dataset and each training case for 256 dimension each)
  disp('Done with all shitty initialization and stuff');
%   
  if n_iters ~= 0
      test_gradient(model, datas.training, wd_coefficient); 
      disp('Done With initial gradient testing')
  end
  
%
%   
%   
%   % optimization
  theta = model_to_theta(model);
  disp('size(theta) is '), disp(size(theta))
  momentum_speed = theta * 0;
  disp('size(momentum) is: '), disp(size(momentum_speed))
  training_data_losses = [];
  validation_data_losses = [];
  if do_early_stopping,
    best_so_far.theta = -1; % this will be overwritten soon
    best_so_far.validation_loss = inf;
    best_so_far.after_n_iters = -1;
    disp('best_so_far is '), disp(best_so_far)
  end

  disp('Done with first optimization initilization and stuff');
  
  % The below code performs optimization and descents for the given
  % iteration
  for optimization_iteration_i = 1:n_iters,
    disp('aaaa Iteration number is:'), disp(optimization_iteration_i)
    model = theta_to_model(theta);
    disp('model is' ), disp(model)
    
    %  get the training data and the target data.   
    training_batch_start = mod((optimization_iteration_i-1) * mini_batch_size, n_training_cases)+1;
    disp('training_batch_start is: '), disp(training_batch_start)
    training_batch.inputs = datas.training.inputs(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    disp('size(training_batch.inputs)  is: '), disp(size(training_batch.inputs))
    training_batch.targets = datas.training.targets(:, training_batch_start : training_batch_start + mini_batch_size - 1);
    disp('size(training_batch.targets) is: '), disp(size(training_batch.targets))
    
    gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient));

    momentum_speed = momentum_speed * momentum_multiplier - gradient;
    theta = theta + momentum_speed * learning_rate;

    model = theta_to_model(theta);
    training_data_losses = [training_data_losses, loss(model, datas.training, wd_coefficient)];
    validation_data_losses = [validation_data_losses, loss(model, datas.validation, wd_coefficient)];
    if do_early_stopping && validation_data_losses(end) < best_so_far.validation_loss,
      best_so_far.theta = theta; % this will be overwritten soon
      best_so_far.validation_loss = validation_data_losses(end);
      best_so_far.after_n_iters = optimization_iteration_i;
    end
    if mod(optimization_iteration_i, round(n_iters/10)) == 0,
      fprintf('After %d optimization iterations, training data loss is %f, and validation data loss is %f\n', optimization_iteration_i, training_data_losses(end), validation_data_losses(end));
    end
  end
  disp('Done with iteration dude, if at all there was any');
  
  if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end % check again, this time with more typical parameters
  if do_early_stopping,
    fprintf('Early stopping: validation loss was lowest after %d iterations. We chose the model that we had then.\n', best_so_far.after_n_iters);
    theta = best_so_far.theta;
  end
  
  disp('Done with the optomization, now we do some reporting.')
  % the optimization is finished. Now do some reporting.
  model = theta_to_model(theta);
  if n_iters ~= 0,
    disp ('their are some iteration....')
    clf;
    hold on;
    plot(training_data_losses, 'b');
    plot(validation_data_losses, 'r');
    legend('training', 'validation');
    ylabel('loss');
    xlabel('iteration number');     
    hold off;
  end
  datas2 = {datas.training, datas.validation, datas.test};
  data_names = {'training', 'validation', 'test'};
  
  % Calculting Loss
  for data_i = 1:3,
    data = datas2{data_i};
    data_name = data_names{data_i};
    fprintf('\nThe loss on the %s data is %f\n', data_name, loss(model, data, wd_coefficient));
    if wd_coefficient~=0,
      fprintf('The classification loss (i.e. without weight decay) on the %s data is %f\n', data_name, loss(model, data, 0));
    end
    fprintf('The classification error rate on the %s data is %f\n', data_name, classification_performance(model, data));
  end
end

function test_gradient(model, data, wd_coefficient)
  base_theta = model_to_theta(model);
  h = 1e-2;
  correctness_threshold = 1e-5;
  analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient));
  % Test the gradient not for every element of theta, because that's a lot of work. Test for only a few elements.
  for i = 1:100,
    test_index = mod(i * 1299721, size(base_theta,1)) + 1; % 1299721 is prime and thus ensures a somewhat random-like selection of indices
    analytic_here = analytic_gradient(test_index);
    theta_step = base_theta * 0;
    theta_step(test_index) = h;
    contribution_distances = [-4:-1, 1:4];
    contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280];
    temp = 0;
    for contribution_index = 1:8,
      temp = temp + loss(theta_to_model(base_theta + theta_step * contribution_distances(contribution_index)), data, wd_coefficient) * contribution_weights(contribution_index);
    end
    fd_here = temp / h;
    diff = abs(analytic_here - fd_here);
    % fprintf('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
    if diff < correctness_threshold, continue; end
    if diff / (abs(analytic_here) + abs(fd_here)) < correctness_threshold, continue; end
    error(sprintf('Theta element #%d, with value %e, has finite difference gradient %e but analytic gradient %e. That looks like an error.\n', test_index, base_theta(test_index), fd_here, analytic_here));
  end
  fprintf('Gradient test passed. That means that the gradient that your code computed is within 0.001%% of the gradient that the finite difference approximation computed, so the gradient calculation procedure is probably correct (not certainly, but probably).\n');
end

function ret = logistic(input)
  ret = 1 ./ (1 + exp(-input));
end

function ret = log_sum_exp_over_rows(a)
%     disp('Inside the ret-log_sum_exp_over_rows function')
  % This computes log(sum(exp(a), 1)) in a numerically stable way
  maxs_small = max(a, [], 1); 
  % Finds the maximum value of each column of the matrix
  maxs_big = repmat(maxs_small, [size(a, 1), 1]); 
  % Create a matrix of same size as that of a but with the biggest columnar element of matrix a 
  ret = log(sum(exp(a - maxs_big), 1)) + maxs_small;
end

function ret = loss(model, data, wd_coefficient)
%     disp('Inside the ret-loss function')
  % model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>. It contains the weights from the input units to the hidden units.
  % model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>. It contains the weights from the hidden units to the softmax units.
  % data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case. 
  % data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, 
  % i.e. one element in every column is 1 and the others are 0.
	 
  % Before we can calculate the loss, we need to calculate a variety of intermediate values, like the state of the hidden units.
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  % The following three lines of code implement the softmax.
  % However, it's written differently from what the lectures say.
  % In the lectures, a softmax is described using an exponential divided by a sum of exponentials.
  % What we do here is exactly equivalent (you can check the math or just check it in practice), 
  % but this is more numerically stable. "Numerically stable" means that this way, there will never be really big numbers involved.
  % The exponential in the lectures can lead to really big numbers, which are fine in mathematical equations, but can lead to all sorts of problems in Octave.
  % Octave isn't well prepared to deal with really large numbers, like the number 10 to the power 1000. Computations with such numbers get unstable, so we avoid them.
  class_normalizer = log_sum_exp_over_rows(class_input); 
  % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
  log_class_prob = class_input - repmat(class_normalizer, [size(class_input, 1), 1]); 
  % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
  class_prob = exp(log_class_prob); 
  % probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
  
  % The below is similar appproch to cross entropy but instead of doing sum
  % we do -mean
%   disp('qpqppqpqpqpqpqpqpqpqpqpqpqpp'),disp(size(log_class_prob .* data.targets))
  classification_loss = -mean(sum(log_class_prob .* data.targets, 1)); 
  % select the right log class probability using that sum; then take the mean over all data cases.
  wd_loss = sum(model_to_theta(model).^2)/2*wd_coefficient; 
  % weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
  % The above weight decay loss formula is equivallent of writting the
  % regularization.
  ret = classification_loss + wd_loss;
end

function ret = d_loss_by_d_model(model, data, wd_coefficient)
    disp('Inside the ret-d_loss_by_d_model function')
  % model.input_to_hid is a matrix of size <number of hidden units> by <number of inputs i.e. 256>
  % model.hid_to_class is a matrix of size <number of classes i.e. 10> by <number of hidden units>
  % data.inputs is a matrix of size <number of inputs i.e. 256> by <number of data cases>. Each column describes a different data case. 
  % data.targets is a matrix of size <number of classes i.e. 10> by <number of data cases>. Each column describes a different data case. It contains a one-of-N encoding of the class, i.e. one element in every column is 1 and the others are 0.

  % The returned object is supposed to be exactly like parameter <model>, i.e. it has fields ret.input_to_hid and ret.hid_to_class. 
  % However, the contents of those matrices are gradients (d loss by d model parameter), instead of model parameters.
	 
  % This is the only function that you're expected to change. 
  % Right now, it just returns a lot of zeros, 
  % which is obviously not the correct output. 
  % Your job is to replace that by a correct computation.
  
  [c, batch_size] = size(data.inputs);
  disp('The Batch Size is: '), disp(batch_size)
  
  
  % Doing Forward Propagation
  disp('')
  disp('Initiating Forward propagation ......................................')
  disp('model is : '), disp(model)
  disp('data is: '), disp(data)
  inp_to_hid_layer = model.input_to_hid * data.inputs;
  hidden_state = logistic(inp_to_hid_layer);
  disp('size(hidden_state) is: '), disp(size(hidden_state))
  disp('size(model.hid_to_class) is: '), disp(size(model.hid_to_class))
  
  hid_to_output_layer =  model.hid_to_class * hidden_state;
  % Doing similar to softmax
  class_normalizer = log_sum_exp_over_rows(hid_to_output_layer);
  log_class_prob = hid_to_output_layer - repmat(class_normalizer, [size(hid_to_output_layer, 1), 1]);
  output_state = exp(log_class_prob); 
  disp('size(output_state) is: '), disp(size(output_state))
  
  
  % Doing Backward propagation
  disp('')
  disp('Initiating Backward propagation .....................................')
  % calculating gradient
  error_deriv = (output_state - data.targets);  
  disp('size of data.targets is: '), disp(size(data.targets))
  disp('size of output_state is: '), disp(size(output_state))
  disp('size(error_deriv) is: '), disp(size(error_deriv)) 
  
  % New hidden_to_output_wghts
  disp('qpqppqpqpqpqpqpqpqpqpqpqpqpp  '),disp(wd_coefficient)
  hid_to_output_wght_gradient = ((1./batch_size) .* (error_deriv * hidden_state')) + (wd_coefficient .*  model.hid_to_class) ;  % wd_coefficient .*
  % (wd_coefficient .* model.hid_to_class)  = gradient of the
  % regularization term
  % (1./batch_size) is done to normalize the matrix as while the cross entropy we
  % take -mean(sum(log_class_prob .* data.targets, 1)). we take the mean
  % which means dividing by the batch size.
  disp('size(hid_to_output_wght_gradient) is: '), disp(size(hid_to_output_wght_gradient))
  
  % New input_to_hidden_wgths
  deriv_e_tot_by_out_h = error_deriv' * model.hid_to_class;  % model.hid_to_class is equivalent to hid_to_output_wgths
  deriv_out_h_by_z_h = hidden_state .* (1-hidden_state);
  back_prop_deriv_1 = deriv_e_tot_by_out_h' .* deriv_out_h_by_z_h;
  input_to_hid_wght_gradient = ((1./batch_size) .* (back_prop_deriv_1 * data.inputs')) + (wd_coefficient .* model.input_to_hid);
  disp('size(deriv_e_tot_by_out_h) is: '), disp(size(deriv_e_tot_by_out_h))
  disp('size(deriv_out_h_by_z_h) is: '), disp(size(deriv_out_h_by_z_h))
  disp('size(back_prop_deriv_1) is: '), disp(size(back_prop_deriv_1))
  disp('size(input_to_hid_wght_gradient) is: '), disp(size(input_to_hid_wght_gradient))
  
  ret.hid_to_class = hid_to_output_wght_gradient;  %   ./ batch_size
  ret.input_to_hid = input_to_hid_wght_gradient; % ./ batch_size;
end

function ret = model_to_theta(model)
%   disp('Inside the ret-model_to_theta function')
  % This function takes a model (or gradient in model form), and turns it into one long vector. See also theta_to_model.
  input_to_hid_transpose = transpose(model.input_to_hid);
  hid_to_class_transpose = transpose(model.hid_to_class);
%   disp('size(input_to_hid_transpose) is: '),disp(size(input_to_hid_transpose))
%   disp ('size(hid_to_class_transpose) is :'),disp(size(hid_to_class_transpose))
  ret = [input_to_hid_transpose(:); hid_to_class_transpose(:)];
end

function ret = theta_to_model(theta)
% disp('Inside the ret-theta_to_model function')
  % This function takes a model (or gradient) in the form of one long vector 
  % (maybe produced by model_to_theta), and restores it to the structure format, 
  % i.e. with fields .input_to_hid and .hid_to_class, both matrices.
  n_hid = size(theta, 1) / (256+10);
  ret.input_to_hid = transpose(reshape(theta(1: 256*n_hid), 256, n_hid));
  ret.hid_to_class = reshape(theta(256 * n_hid + 1 : size(theta,1)), n_hid, 10).';
end

function ret = initial_model(n_hid)
%   disp('Inside the ret-initial_model function')
  n_params = (256+10) * n_hid;  % num of parameters
  as_row_vector = cos(0:(n_params-1));  % 1 x n_params
  ret = theta_to_model(as_row_vector(:) * 0.1); 
  % We don't use random initialization, for this assignment. This way, everybody will get the same results.
end

function ret = classification_performance(model, data)
% disp('Inside the ret-classification_performance function')
  % This returns the fraction of data cases that is incorrectly classified by the model.
  hid_input = model.input_to_hid * data.inputs; % input to the hidden units, i.e. before the logistic. size: <number of hidden units> by <number of data cases>
  hid_output = logistic(hid_input); % output of the hidden units, i.e. after the logistic. size: <number of hidden units> by <number of data cases>
  class_input = model.hid_to_class * hid_output; % input to the components of the softmax. size: <number of classes, i.e. 10> by <number of data cases>
  
  [dump, choices] = max(class_input); % choices is integer: the chosen class, plus 1.
  [dump, targets] = max(data.targets); % targets is integer: the target class, plus 1.
  ret = mean(double(choices ~= targets));
end


% addpath('/Users/sam/All-Program/App-DataSet/Study/Coursera/assignment3/')
% addpath('/Users/sam/All-Program/App/Study/Coursera/assignment3')