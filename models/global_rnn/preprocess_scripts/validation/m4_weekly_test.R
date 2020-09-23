# Using a seasonal lag to deal with seasonality

library(forecast)

# Define paths and initial variables
BASE_DIR <- "WeeklyForecasting"
DATA_DIR <- file.path(BASE_DIR, "datasets")
OUTPUT_DIR <- file.path(DATA_DIR, "preprocessed_data")
dataset_name <- "m4_weekly_dataset.txt"
output_file_prefix <- "m4_weekly_test_"
input_size <- 53
max_forecast_horizon <- 13
phase <- "validation"

# Create the output directory if it does not exist
dir.create(file.path(OUTPUT_DIR), showWarnings = FALSE)

df_train <- readLines(file.path(DATA_DIR, dataset_name))
df_train <- strsplit(df_train, ',')

OUTPUT_PATH <- paste(OUTPUT_DIR, output_file_prefix, sep = '/')
OUTPUT_PATH <- paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
OUTPUT_PATH <- paste(OUTPUT_PATH, 'i', input_size, sep = '')
OUTPUT_PATH <- paste(OUTPUT_PATH, '_', phase, sep = '')
OUTPUT_PATH <- paste(OUTPUT_PATH, 'txt', sep = '.')

for (idr in 1 : length(df_train)) {
    print(idr)

    time_series_data <- unlist(df_train[idr], use.names = FALSE)
    time_series_data <- as.numeric(time_series_data)

    # Mean Normalisation
    time_series_mean <- mean(time_series_data)
    time_series_data <- time_series_data / time_series_mean

    # Log Transformation
    time_series_log <- log(time_series_data)
    time_series_length <- length(time_series_log)

     # Remove the last set of points where the size is equal to forecast horizon
    time_series_length <- time_series_length - max_forecast_horizon
    time_series_log <- time_series_log[1 : time_series_length]

    input_windows <- embed(time_series_log[1 : (time_series_length)], input_size)[, input_size : 1]

    # Per-window Normalisation
    meanvalues <- rowMeans(input_windows)
    input_windows <- input_windows - meanvalues

    # Creating the final matrix
    sav_df <- matrix(NA, ncol = (1 + input_size + 3), nrow = nrow(input_windows))
    sav_df <- as.data.frame(sav_df)
    sav_df[, 1] <- paste(idr - 1, '|i', sep = '')
    sav_df[, 2 : (input_size + 1)] <- input_windows
    sav_df[, (input_size + 2)] <- '|#'
    sav_df[, (input_size + 3)] <- time_series_mean
    sav_df[, (input_size + 4)] <- meanvalues

    write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}