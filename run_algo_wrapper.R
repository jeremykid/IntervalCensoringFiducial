#!/usr/bin/env Rscript
# R wrapper for algorithm.R - generates survival function estimates with confidence intervals
# Usage: Rscript run_algo_wrapper.R <input_csv> <output_csv> [seed] [mfid] [mburn] [alpha]

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_algo_wrapper.R <input_csv> <output_csv> [seed] [mfid] [mburn] [alpha]")
}

in_csv <- args[1]
out_csv <- args[2]
seed <- ifelse(length(args) >= 3, as.integer(args[3]), 123)
mfid <- ifelse(length(args) >= 4, as.integer(args[4]), 1000)
mburn <- ifelse(length(args) >= 5, as.integer(args[5]), 100)
alpha <- ifelse(length(args) >= 6, as.numeric(args[6]), 0.05)

cat("=== Interval Censoring Fiducial Algorithm Wrapper ===\n")
cat("Input CSV:", in_csv, "\n")
cat("Output CSV:", out_csv, "\n")
cat("Random seed:", seed, "\n")
cat("mfid (fiducial samples):", mfid, "\n")
cat("mburn (burn-in samples):", mburn, "\n")
cat("alpha (confidence level):", alpha, "\n\n")

# Read input data
df <- read.csv(in_csv)
if (!all(c("left", "right") %in% colnames(df))) {
  stop("Input CSV must contain 'left' and 'right' columns")
}

l <- df$left
r <- df$right
n <- length(l)

cat("Data loaded: n =", n, "observations\n")
cat("Left range: [", min(l), ",", max(l), "]\n")
cat("Right range: [", min(r[is.finite(r)]), ",", max(r[is.finite(r)]), "] (excluding Inf)\n\n")

# Set up grid parameters
# grid.high should not include Inf values
finite_r <- r[is.finite(r)]
if (length(finite_r) == 0) {
  stop("All right censoring values are Inf - cannot determine grid.high")
}

grid.low <- min(l)
grid.high <- max(finite_r)
ngrid <- 100  # number of points in the fiducial grid

# testgrid is where we want final survival function estimates
# Use a finer grid for smoother curves
testgrid <- seq(grid.low, grid.high, length.out = 200)

cat("Grid setup:\n")
cat("  grid.low =", grid.low, "\n")
cat("  grid.high =", grid.high, "\n")
cat("  ngrid =", ngrid, "\n")
cat("  testgrid points =", length(testgrid), "\n\n")

# Set random seed for reproducibility
set.seed(seed)

# Source the algorithm (assumes it's in the same directory)
# Get script directory - works with Rscript
initial_options <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", initial_options, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- sub("^--file=", "", file_arg)
  script_dir <- dirname(script_path)
} else {
  script_dir <- getwd()
}

cat("Working directory:", getwd(), "\n")
cat("Script directory:", script_dir, "\n")

# Change to script directory to find algorithm.R and LinInterpolation.R
old_dir <- getwd()
setwd(script_dir)

cat("Sourcing algorithm.R...\n")
source("algorithm.R")

cat("\n=== Algorithm completed ===\n")
cat("FiducialMidLine1 dimensions:", dim(FiducialMidLine1), "\n")
cat("  (should be", length(testgrid), "x", mfid, ")\n")
cat("point_li length:", length(point_li), "\n\n")

# Calculate survival function estimates
cat("Calculating survival function statistics...\n")

# FiducialMidLine1 is ntestgrid x mfid matrix
# Each row corresponds to a time point, each column is a fiducial sample
# Values are CDF estimates F(t)

# Calculate mean and quantiles of CDF
F_mean <- apply(FiducialMidLine1, 1, mean)
F_low <- apply(FiducialMidLine1, 1, quantile, probs = alpha/2)
F_high <- apply(FiducialMidLine1, 1, quantile, probs = 1 - alpha/2)

# Convert to survival function S(t) = 1 - F(t)
S_hat <- 1 - F_mean
S_low <- 1 - F_high  # Note the flip: lower CI of S = 1 - upper CI of F
S_high <- 1 - F_low

# Also calculate point estimate from point_li
S_point_li <- 1 - point_li

# Create output dataframe
output_df <- data.frame(
  time = testgrid,
  S_hat = S_hat,
  S_low = S_low,
  S_high = S_high,
  S_point_li = S_point_li,
  F_mean = F_mean,
  F_low = F_low,
  F_high = F_high
)

# Restore original directory
setwd(old_dir)

# Save output
write.csv(output_df, out_csv, row.names = FALSE)
cat("Results saved to:", out_csv, "\n\n")

# Print debug information
cat("=== Debug Information ===\n")
cat("First few rows of output:\n")
print(head(output_df, 10))
cat("\nLast few rows of output:\n")
print(tail(output_df, 10))
cat("\nSummary statistics:\n")
cat("  S_hat range: [", min(S_hat), ",", max(S_hat), "]\n")
cat("  S_low range: [", min(S_low), ",", max(S_low), "]\n")
cat("  S_high range: [", min(S_high), ",", max(S_high), "]\n")
cat("\nFirst 10 point_li values:", head(point_li, 10), "\n")
cat("First 10 F_mean values:", head(F_mean, 10), "\n")

cat("\n=== Wrapper completed successfully ===\n")
