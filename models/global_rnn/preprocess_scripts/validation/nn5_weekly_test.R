# Use fourier terms to deal with seasonality

library(forecast)

# Define paths and initial variables
BASE_DIR <- "WeeklyForecasting"
DATA_DIR <- file.path(BASE_DIR, "datasets")
OUTPUT_DIR <- file.path(DATA_DIR, "preprocessed_data")
dataset_name <- "nn5_weekly_dataset.txt"
output_file_prefix <- "nn5_weekly_test_"
k_val <- 5
input_size <- 10
full_input_size <- 20 # input size after adding fourier terms
max_forecast_horizon <- 8
seasonality_period <- 365.25/7
phase <- "validation"

# Create the output directory if it does not exist
dir.create(file.path(OUTPUT_DIR), showWarnings = FALSE)

df_train <- readLines(file.path(DATA_DIR, dataset_name))
df_train <- strsplit(df_train, ',')

# Calculate fourier terms for the required k value
# This dataset does not require aligning fourier terms across time as all series lengths are same
# You may need to align fourier terms according to the characteristics of dataset
time_series_data <- unlist(df_train[1], use.names = FALSE)
time_series_data <- as.numeric(time_series_data)
fourier_terms <- fourier(ts(time_series_data, frequency = seasonality_period), K = k_val)

OUTPUT_PATH <- paste(OUTPUT_DIR, output_file_prefix, sep = '/')
OUTPUT_PATH <- paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
OUTPUT_PATH <- paste(OUTPUT_PATH, 'i', full_input_size, sep = '')
OUTPUT_PATH <- paste(OUTPUT_PATH, '_', phase, sep = '')
OUTPUT_PATH <- paste(OUTPUT_PATH, 'txt', sep = '.')

for (idr in 1 : length(df_train)) {
    print(idr)
    time_series_data <- unlist(df_train[idr], use.names = FALSE)
    time_series_data <- as.numeric(time_series_data)
    time_series_mean <- mean(time_series_data)

    # Mean normalisation
    time_series_data <- time_series_data / time_series_mean

    # Log Transformation
    time_series_log <- log(time_series_data)
    time_series_length <- length(time_series_log)

    # Remove the last set of points where the size is equal to forecast horizon
    time_series_length <- time_series_length - max_forecast_horizon
    time_series_log <- time_series_log[1 : time_series_length]

    required_fourier_series <- fourier_terms

    input_windows <- embed(time_series_log[1 : (time_series_length)], input_size)[, input_size : 1]

    sin_indexes <- seq(1, (k_val * 2), 2)

    for (val in sin_indexes) {
        sin_terms <- required_fourier_series[, val]
        cos_terms <- required_fourier_series[, (val + 1)]

        sin_windows <- embed(sin_terms[1 : (time_series_length)], input_size)[, input_size : 1]
        cos_windows <- embed(cos_terms[1 : (time_series_length)], input_size)[, input_size : 1]

        sin_windows <- sin_windows[, c(input_size)]
        cos_windows <- cos_windows[, c(input_size)]

        assign(paste0("seasonality_sin_windows_", val), sin_windows)
        assign(paste0("seasonality_cos_windows_", (val + 1)), cos_windows)
    }

    # Per-window normalisation
    meanvalues <- rowMeans(input_windows)
    input_windows <- input_windows - meanvalues

    # Creating final test matrix.
    # You may need to modify this according to the number of fourier terms you use
    sav_df <- matrix(NA, ncol = (11 + input_size + 3), nrow = nrow(input_windows))
    sav_df <- as.data.frame(sav_df)
    sav_df[, 1] <- paste(idr - 1, '|i', sep = '')
    sav_df[, 2] <- seasonality_sin_windows_1
    sav_df[, 3] <- seasonality_cos_windows_2
    sav_df[, 4] <- seasonality_sin_windows_3
    sav_df[, 5] <- seasonality_cos_windows_4
    sav_df[, 6] <- seasonality_sin_windows_5
    sav_df[, 7] <- seasonality_cos_windows_6
    sav_df[, 8] <- seasonality_sin_windows_7
    sav_df[, 9] <- seasonality_cos_windows_8
    sav_df[, 10] <- seasonality_sin_windows_9
    sav_df[, 11] <- seasonality_cos_windows_10
    sav_df[, 12 : (input_size + 10 + 1)] <- input_windows
    sav_df[, (input_size + 10 + 2)] <- '|#'
    sav_df[, (input_size + 10 + 3)] <- time_series_mean
    sav_df[, (input_size + 10 + 4)] <- meanvalues

    write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}