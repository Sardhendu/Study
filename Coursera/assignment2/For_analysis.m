batchsize=2
numhid2=3
momentum = 0.9
learning_rate = 0.1
hid_bias = zeros(3,1)
hid_bias_delta = zeros(3,1)
output_bias = zeros(2,1)
output_bias_delta = zeros(2,1)
embed_to_hid_weights_delta = zeros(2,3)
hid_to_output_weights_delta = zeros(3,2)

expanded_target_batch = [1,0;0,1]
embedding_layer_state = [2,4;3,5]
embed_to_hid_weights = [-0.75713223,  1.38995981, -1.54961336;0.51295179, -0.05542428, -0.18984629]
hid_to_output_weights = [-0.75713223,  1.38995981; -1.54961336,  0.51295179;-0.05542428, -0.18984629]


% Forward Pass
inputs_to_hidden_units = embed_to_hid_weights' * embedding_layer_state + ...
  repmat(hid_bias, 1, batchsize);
hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));  
inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  ...
    repmat(output_bias, 1, batchsize);
inputs_to_softmax = inputs_to_softmax...
  - repmat(max(inputs_to_softmax), vocab_size, 1);
    output_layer_state = exp(inputs_to_softmax);

output_layer_state = output_layer_state ./ repmat(...
  sum(output_layer_state, 1), vocab_size, 1);

error_deriv = output_layer_state - expanded_target_batch


% Back Propagate
hid_to_output_weights_gradient =  hidden_layer_state * error_deriv';
    output_bias_gradient = sum(error_deriv, 2);
    back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) ...
      .* hidden_layer_state .* (1 - hidden_layer_state);
        
embed_to_hid_weights_gradient = ...
        embedding_layer_state * back_propagated_deriv_1';
    
hid_bias_gradient = sum(back_propagated_deriv_1, 2);

embed_to_hid_weights_delta = ...
      momentum .* embed_to_hid_weights_delta + ...
      embed_to_hid_weights_gradient ./ batchsize;
embed_to_hid_weights = embed_to_hid_weights...
      - learning_rate * embed_to_hid_weights_delta;

hid_to_output_weights_delta = ...
      momentum .* hid_to_output_weights_delta + ...
      hid_to_output_weights_gradient ./ batchsize;
hid_to_output_weights = hid_to_output_weights...
      - learning_rate * hid_to_output_weights_delta;

hid_bias_delta = momentum .* hid_bias_delta + ...
      hid_bias_gradient ./ batchsize;
hid_bias = hid_bias - learning_rate * hid_bias_delta;

output_bias_delta = momentum .* output_bias_delta + ...
      output_bias_gradient ./ batchsize;
output_bias = output_bias - learning_rate * output_bias_delta;

