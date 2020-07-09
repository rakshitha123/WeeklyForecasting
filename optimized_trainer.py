# Use this file to run RNN for validation phase
# It will load the hyperparameters which were identified by using the previous run of generic_model_trainer.py for the test phase
# This will train the RNN from the loaded hyperparameters and then will connect to other scripts which will do forecasting and re-scaling
# Make sure to provide the hyperparameters in json format
import tensorflow as tf
import argparse
import json
from generic_model_tester import testing
from models.global_rnn.external_packages import cocob_optimizer  # import the cocob optimizer

LSTM_USE_PEEPHOLES = True
BIAS = False

learning_rate = 0.0

# function to create the optimizer
def adagrad_optimizer_fn(total_loss):
    return tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_loss)

def adam_optimizer_fn(total_loss):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

def cocob_optimizer_fn(total_loss):
    return cocob_optimizer.COCOB().minimize(loss=total_loss)

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Train different forecasting models")
    argument_parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    argument_parser.add_argument('--contain_zero_values', required=True, help='Whether the dataset contains zero values(0/1)')
    argument_parser.add_argument('--address_near_zero_instability', required=False, help='Whether to use a custom SMAPE function to address near zero instability(0/1). Default is 0')
    argument_parser.add_argument('--integer_conversion', required=False, help='Whether to convert the final forecasts to integers(0/1). Default is 0')
    argument_parser.add_argument('--binary_train_file_test_mode', required=True, help='The tfrecords file for train dataset in the testing mode')
    argument_parser.add_argument('--binary_test_file_test_mode', required=True, help='The tfrecords file for test dataset in the testing mode')
    argument_parser.add_argument('--txt_test_file', required=True, help='The txt file for test dataset')
    argument_parser.add_argument('--actual_results_file', required=True, help='The txt file of the actual results')
    argument_parser.add_argument('--original_data_file', required=True, help='The txt file of the original dataset')
    argument_parser.add_argument('--cell_type', required=False, help='The cell type of the RNN(LSTM/GRU/RNN). Default is LSTM')
    argument_parser.add_argument('--input_size', required=False, help='The input size of the moving window. Default is 0')
    argument_parser.add_argument('--hyperparameter_tuning', required=True, help='The method for hyperparameter tuning(bayesian/smac)')
    argument_parser.add_argument('--seasonality_period', required=True, help='The seasonality period of the time series')
    argument_parser.add_argument('--forecast_horizon', required=True, help='The forecast horizon of the dataset')
    argument_parser.add_argument('--optimizer', required=True, help='The type of the optimizer(cocob/adam/adagrad...)')
    argument_parser.add_argument('--hyperparameters', required=True, help='File path containing hyperparameters')
    argument_parser.add_argument('--input_format', required=True, help='Input format(moving_window/non_moving_window)')
    argument_parser.add_argument('--without_stl_decomposition', required=False, help='Whether not to use stl decomposition(0/1). Default is 0')
    argument_parser.add_argument('--with_truncated_backpropagation', required=False, help='Whether not to use truncated backpropagation(0/1). Default is 0')
    argument_parser.add_argument('--with_accumulated_error', required=False, help='Whether to accumulate errors over the moving windows. Default is 0')
    argument_parser.add_argument('--seed', required=True, help='Integer seed to use as the random seed')

    # parse the user arguments
    args = argument_parser.parse_args()

    with open(args.hyperparameters, "r") as read_file:
        optimized_params = json.load(read_file)

    dataset_name = args.dataset_name
    contain_zero_values = int(args.contain_zero_values)

    if args.input_size:
        input_size = int(args.input_size)
    else:
        input_size = 0

    output_size = int(args.forecast_horizon)
    optimizer = args.optimizer
    input_format = args.input_format
    seed = int(args.seed)

    if args.without_stl_decomposition:
        without_stl_decomposition = bool(int(args.without_stl_decomposition))
    else:
        without_stl_decomposition = False

    if args.with_truncated_backpropagation:
        with_truncated_backpropagation = bool(int(args.with_truncated_backpropagation))
    else:
        with_truncated_backpropagation = False

    if args.cell_type:
        cell_type = args.cell_type
    else:
        cell_type = "LSTM"

    if args.with_accumulated_error:
        with_accumulated_error = bool(int(args.with_accumulated_error))
    else:
        with_accumulated_error = False

    if args.address_near_zero_instability:
        address_near_zero_instability = bool(int(args.address_near_zero_instability))
    else:
        address_near_zero_instability = False

    if args.integer_conversion:
        integer_conversion = bool(int(args.integer_conversion))
    else:
        integer_conversion = False

    model_identifier = dataset_name + "_rnn_validation"

    print("Model Training Started for {}".format(model_identifier))

    # select the optimizer
    if optimizer == "cocob":
        optimizer_fn = cocob_optimizer_fn
    elif optimizer == "adagrad":
        optimizer_fn = adagrad_optimizer_fn
    elif optimizer == "adam":
        optimizer_fn = adam_optimizer_fn

    optimized_configuration = {
        'num_hidden_layers': optimized_params['num_hidden_layers'],
        'cell_dimension': optimized_params['cell_dimension'],
        'l2_regularization': optimized_params['l2_regularization'],
        'gaussian_noise_stdev': optimized_params['gaussian_noise_stdev'],
        'random_normal_initializer_stdev': optimized_params['random_normal_initializer_stdev'],
        'minibatch_size': optimized_params['minibatch_size'],
        'max_epoch_size': optimized_params['max_epoch_size'],
        'max_num_epochs': optimized_params['max_num_epochs'],
        'learning_rate': ''
    }

    testing(args, optimized_configuration, "validation")