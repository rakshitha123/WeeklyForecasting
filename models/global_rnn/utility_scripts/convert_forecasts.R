library(smooth)
library(MASS)

root_directory <- "WeeklyForecasting"

source(file.path(root_directory, "utils", "error_calculator.R", fsep = "/"))
source(file.path(root_directory, "utils", "forecast_helper.R", fsep = "/"))

args <- commandArgs(trailingOnly = TRUE)
rnn_forecast_file_path <- args[1]
errors_directory <- args[2]
processed_forecasts_directory <- args[3]
errors_file_name <- args[4]
txt_test_file_name <- args[5]
actual_results_file_name <- args[6]
original_data_file_name <- args[7]
input_size <- as.numeric(args[8])
output_size <- as.numeric(args[9])
contain_zero_values <- as.numeric(args[10])
address_near_zero_insability <- as.numeric(args[11])
integer_conversion <- as.numeric(args[12])
seasonality_period <- as.numeric(args[13])
without_stl_decomposition <- as.numeric(args[14])
phase <- args[15]

# loading original dataset and actual results
training_test_sets <- create_traing_test_sets(original_data_file_name, actual_results_file_name)
original_dataset <- training_test_sets[[1]]
actual_results <- training_test_sets[[2]]

# text test data file name
txt_test_df <- read.csv(file = txt_test_file_name, sep = " ", header = FALSE)

# rnn_forecasts file name
forecasts_df <- read.csv(rnn_forecast_file_path, header = F, sep = ",")

# persisting the final forecasts
processed_forecasts_file <- paste(processed_forecasts_directory, errors_file_name, '.txt', sep = "")

# take the transpose of the dataframe
value <- t(txt_test_df[1])

indexes <- length(value) - match(unique(value), rev(value)) + 1

uniqueindexes <- unique(indexes)

converted_forecasts_df <- NULL
converted_forecasts_matrix <- matrix(nrow = nrow(forecasts_df), ncol = output_size)

for (k in 1 : nrow(forecasts_df)) {
    one_ts_forecasts <- as.numeric(forecasts_df[k,])
    finalindex <- uniqueindexes[k]
    one_line_test_data <- as.numeric(txt_test_df[finalindex,])
    level_value <- one_line_test_data[input_size + 3]

    if (without_stl_decomposition == 1) {
        per_window_mean_value = one_line_test_data[input_size + 4]
        converted_forecasts_df <- exp(one_ts_forecasts + per_window_mean_value)
    }else {
        seasonal_values <- one_line_test_data[(input_size + 4) : (3 + input_size + output_size)]
        converted_forecasts_df <- exp(one_ts_forecasts + level_value + seasonal_values)
    }

    if (contain_zero_values == 1)
        converted_forecasts_df <- converted_forecasts_df - 1

    if (without_stl_decomposition == 1)
        converted_forecasts_df <- converted_forecasts_df * level_value

    if (integer_conversion == 1)
        converted_forecasts_df <- round(converted_forecasts_df)

    converted_forecasts_df[converted_forecasts_df < 0] <- 0 # to make all forecasts positive
    converted_forecasts_matrix[k,] <- converted_forecasts_df
}

# persisting the converted forecasts
write.matrix(converted_forecasts_matrix, processed_forecasts_file, sep = " ")

if(phase == "test")
    calculate_errors(converted_forecasts_matrix, actual_results, original_dataset, seasonality_period, file.path(errors_directory, errors_file_name), as.logical(address_near_zero_insability))

print("Finished")
