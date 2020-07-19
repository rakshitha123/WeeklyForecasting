from models.global_rnn.tfrecords_handler.moving_window.tfrecord_writer import TFRecordWriter
import os

output_path = "../../../../datasets/preprocessed_data/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Change parameters accordingly for new datasets      
if __name__ == '__main__':
    tfrecord_writer = TFRecordWriter(
        input_size = 20,
        output_size = 8,
        train_file_path = output_path + 'nn5_weekly_8i20_test.txt',
        validate_file_path = output_path + 'nn5_weekly_8i20v_test.txt',
        test_file_path = output_path + 'nn5_weekly_test_8i20_test.txt',
        binary_train_file_path = output_path + 'nn5_weekly_8i20_test.tfrecords',
        binary_validation_file_path = output_path + 'nn5_weekly_8i20v_test.tfrecords',
        binary_test_file_path = output_path + 'nn5_weekly_test_8i20_test.tfrecords'
    )

    tfrecord_writer.read_text_data()
    tfrecord_writer.write_train_data_to_tfrecord_file()
    tfrecord_writer.write_validation_data_to_tfrecord_file()
    tfrecord_writer.write_test_data_to_tfrecord_file()

