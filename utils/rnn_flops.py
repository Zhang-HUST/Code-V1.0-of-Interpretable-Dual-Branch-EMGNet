def get_layer_names(model_type):
    if model_type == 'LSTMModel':
        layer_names = ['LSTM1', 'LSTM2', 'LSTM3']
    elif model_type in ['MyoNet', 'CNN-LSTM', 'Sinc-LSTM']:
        layer_names = ['LSTM1', 'LSTM2']
    elif model_type in ['CNN-GRU', 'Sinc-GRU']:
        layer_names = ['GRU1', 'GRU2']
    elif model_type in ['CNN-BiLSTM', 'Sinc-BiLSTM']:
        layer_names = ['BiLSTM1', 'BiLSTM2']
    elif model_type in ['CNN-BiGRU', 'Sinc-BiGRU']:
        layer_names = ['BiGRU1', 'BiGRU2']
    else:
        raise ValueError('Unsupported model_type!')

    return layer_names


def calculate_lstm_flops(input_size, hidden_size, num_time_steps):
    # Calculate FLOPs for a single LSTM cell
    lstm_flops_per_step = 4 * (input_size + hidden_size) * hidden_size
    # FLOPs that extend to the whole sequence
    total_lstm_flops = num_time_steps * lstm_flops_per_step

    return total_lstm_flops


def calculate_gru_flops(input_size, hidden_size, num_time_steps):
    # Calculate FLOPs for a single GRU cell
    gru_flops_per_step = 3 * (input_size + hidden_size) * hidden_size
    # FLOPs that extend to the whole sequence
    total_gru_flops = num_time_steps * gru_flops_per_step

    return total_gru_flops


def get_rnn_flops(model, model_type):
    layer_names = get_layer_names(model_type)
    total_flops = 0
    for layer_name in layer_names:
        layer = model.get_layer(name=layer_name)
        param_count = layer.count_params()
        print(f"Layer '{layer_name}' :, Parameter Count: {param_count}")
        input_shape = layer.input_shape
        print(f"Layer '{layer_name}' :, Input Shape: {input_shape}")
        output_shape = layer.output_shape
        print(f"Layer '{layer_name}' :, Output shape Shape: {output_shape}")
        input_size, hidden_size, num_time_steps = input_shape[-1], output_shape[-1], input_shape[1]
        # print('input_size: ', input_size, ' , hidden_size: ', hidden_size, ' , num_time_steps: ', num_time_steps)
        # 判断一个字符串中是否有另一个字符串
        if 'LSTM' in layer_name:
            flops = calculate_lstm_flops(input_size, hidden_size, num_time_steps)
        elif 'GRU' in layer_name:
            flops = calculate_gru_flops(input_size, hidden_size, num_time_steps)
        else:
            raise ValueError('Unsupported layer_name!')
        print(f"Layer '{layer_name}' :, Flops: {flops}")
        total_flops = total_flops + flops
    print(f"RNN layers in '{model_type}' :, Total flops: {total_flops}")

    return total_flops
