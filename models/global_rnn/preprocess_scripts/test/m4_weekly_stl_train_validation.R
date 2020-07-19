# Use STL decomposition to deal with seasonality

library(forecast)

# Define paths and initial variables
BASE_DIR <- "WeeklyForecasting"
DATA_DIR <- file.path(BASE_DIR, "datasets")
OUTPUT_DIR <- file.path(DATA_DIR, "preprocessed_data")
dataset_name <- "m4_weekly_dataset.txt"
output_file_prefix <- "m4_weekly_"
input_size <- 16
max_forecast_horizon <- 13
seasonality_period <- 365.25/7
phase <- "test"

# Create the output directory if it does not exist
dir.create(file.path(OUTPUT_DIR), showWarnings = FALSE)

df_train <- readLines(file.path(DATA_DIR, dataset_name))
df_train <- strsplit(df_train, ',')

for (validation in c(TRUE, FALSE)) {
    for (idr in 1 : length(df_train)) {
        print(idr)
        OUTPUT_PATH <- paste(OUTPUT_DIR, output_file_prefix, sep = '/')
        OUTPUT_PATH <- paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
        OUTPUT_PATH <- paste(OUTPUT_PATH, 'i', input_size, sep = '')

        if (validation) {
            OUTPUT_PATH <- paste(OUTPUT_PATH, 'v', sep = '')
        }

        OUTPUT_PATH <- paste(OUTPUT_PATH, '_', phase, sep = '')
        OUTPUT_PATH <- paste(OUTPUT_PATH, 'txt', sep = '.')

        time_series_data <- unlist(df_train[idr], use.names = FALSE)
        time_series_data <- as.numeric(time_series_data)

        # Log Transformation
        time_series_log <- log(time_series_data)
        time_series_length <- length(time_series_log)

        if (! validation) {
            time_series_length <- time_series_length - max_forecast_horizon
            time_series_log <- time_series_log[1 : time_series_length]
        }

        stl_result <- tryCatch({
            sstl <- stl(ts(time_series_log, frequency = seasonality_period), "period")
            seasonal_vect <- as.numeric(sstl$time.series[, 1])
            levels_vect <- as.numeric(sstl$time.series[, 2])
            values_vect <- as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3])# this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
            cbind(seasonal_vect, levels_vect, values_vect)
        }, error = function(e) {
            seasonal_vect <- rep(0, time_series_length)#stl() may fail, and then we would go on with the seasonality vector=0
            levels_vect <- time_series_log
            values_vect <- time_series_log
            cbind(seasonal_vect, levels_vect, values_vect)
        })

        input_windows <- embed(stl_result[1 : (time_series_length - max_forecast_horizon), 3], input_size)[, input_size : 1]
        output_windows <- embed(stl_result[- (1 : input_size) , 3], max_forecast_horizon)[, max_forecast_horizon : 1]

        # Trend normalization
        level_values <- stl_result[input_size : (time_series_length - max_forecast_horizon), 2]
        input_windows <- input_windows - level_values
        output_windows <- output_windows - level_values

        # Creating final training and validation matrices.
        if (validation) {
            # create the seasonality metadata
            seasonality_windows <- embed(stl_result[- (1 : input_size) , 1], max_forecast_horizon)[, max_forecast_horizon : 1]
            sav_df <- matrix(NA, ncol = (1 + input_size + 3 + max_forecast_horizon*2), nrow = nrow(input_windows))
            sav_df <- as.data.frame(sav_df)
            sav_df[, 1] <- paste(idr - 1, '|i', sep = '')
            sav_df[, 2 : (input_size + 1)] <- input_windows
            sav_df[, (input_size + 2)] <- '|o'
            sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] <- output_windows
            sav_df[, (input_size + max_forecast_horizon + 3)] <- '|#'
            sav_df[, (input_size + max_forecast_horizon + 4)] <- level_values
            sav_df[, (input_size + max_forecast_horizon + 5):ncol(sav_df)] <- seasonality_windows
        }else {
            sav_df <- matrix(NA, ncol = (1 + input_size + 1 + max_forecast_horizon), nrow = nrow(input_windows))
            sav_df <- as.data.frame(sav_df)
            sav_df[, 1] <- paste(idr - 1, '|i', sep = '')
            sav_df[, 2 : (input_size + 1)] <- input_windows
            sav_df[, (input_size + 2)] <- '|o'
            sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] <- output_windows
        }

        write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    }
}
