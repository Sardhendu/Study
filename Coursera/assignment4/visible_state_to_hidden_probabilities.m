function hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by 
% <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of hidden units> by 
% <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the visible units, 
% and returns the activation probabilities of the hidden units conditional 
% on those states.
    #disp(size(rbm_w));
    #disp(size(visible_state));
    
    #disp(rbm_w(1:5,1:5))
    #disp(rbm_w(1:5,:))
    
    hidden_probability = zeros(100,1);
    [row,col] = size(rbm_w);
    prob_h_given_v = zeros(row, col);
    for c = 1:col
      #disp(sum(rbm_w(:,c)));
      prob_h_given_v(:,c) = rbm_w(:,c)/sum(rbm_w(:,c));
    endfor
    
    hidden_probability = prob_h_given_v * visible_state;
    #disp(mean(mean(hiddenState)));
    #disp(sum(sum(hiddenState)));
    
    
    #error('not yet implemented');
endfunction


# describe_matrix(visible_state_to_hidden_probabilities(test_rbm_w, data_1_case))
# test_rbm_w : 100 x 256
# data_1_case : 256 x 1


# a = [1,2,3;4,5,6;7,8,9;1,2,1]

# b = zeros(4,3)
