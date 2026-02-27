# =============================================================================
# ms_garch.R — Markov-Switching GARCH via MSGARCH
#
# Part 1: Demo on simulated returns with injected high-vol regime.
# Part 2: Applied to tick-level mid-price returns for EURUSD, USDZAR, XAUUSD.
#
# Prerequisites:
#   install.packages(c("MSGARCH", "arrow"))
#
# Run from RStudio with working directory set to code/R/
# or source("code/R/ms_garch.R") from project root.
# =============================================================================

library(MSGARCH)
library(arrow)

# --- Path setup (relative to this script's location) -------------------------
# Works whether you source() from project root or run in RStudio with
# working directory = code/R/
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)
# Navigate to project root: code/R/ -> code/ -> project root
project_root <- normalizePath(file.path(script_dir, "..", ".."), mustWork = FALSE)
data_dir     <- file.path(project_root, "code", "data", "processed")
models_dir   <- file.path(project_root, "code", "plots", "models")
plotly_dir   <- file.path(project_root, "code", "plots", "plotly")

dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plotly_dir, showWarnings = FALSE, recursive = TRUE)

cat("Project root:", project_root, "\n")
cat("Data dir:    ", data_dir, "\n")
cat("Output dirs: ", models_dir, "\n")
cat("             ", plotly_dir, "\n\n")


# =============================================================================
# Part 1: Simulated Returns
# =============================================================================
cat("=== Part 1: Simulated Returns — MS-GARCH ===\n")

set.seed(42)
n <- 1000
returns <- rnorm(n, mean = 0, sd = 1)
# Inject high volatility regime between days 400-600
returns[400:600] <- rnorm(201, mean = 0, sd = 3)

# Create MS-GARCH specification: 2-regime sGARCH, Normal innovations
spec <- CreateSpec(
  variance.spec    = list(model = c("sGARCH", "sGARCH")),
  distribution.spec = list(distribution = c("norm", "norm")),
  switch.spec      = list(do.mix = FALSE)
)

cat("  Fitting MS-GARCH on simulated data...\n")
fit_sim <- FitML(spec = spec, data = returns)

cat("  Summary:\n")
print(summary(fit_sim))

# Extract conditional volatility
vol_sim <- Volatility(fit_sim)

# Save fit results to CSV for Python plotting
sim_df <- data.frame(
  day     = 1:n,
  returns = returns,
  cond_vol = as.numeric(vol_sim)
)
sim_csv <- file.path(models_dir, "ms_garch_simulated.csv")
write.csv(sim_df, sim_csv, row.names = FALSE)
cat("  saved", basename(sim_csv), "\n")

# Quick R plot (PDF)
pdf_path <- file.path(models_dir, "ms_garch_simulated.pdf")
pdf(pdf_path, width = 10, height = 6)
par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))
plot(returns, type = "l", col = "steelblue", lwd = 0.7,
     main = "Simulated Returns (high-vol injected days 400-600)",
     ylab = "Return", xlab = "")
abline(v = c(400, 600), col = "red", lty = 2)
plot(as.numeric(vol_sim), type = "l", col = "darkorange", lwd = 1.2,
     main = "MS-GARCH Conditional Volatility",
     ylab = "Volatility", xlab = "Day")
dev.off()
cat("  saved", basename(pdf_path), "\n")


# =============================================================================
# Part 2: Tick Data — EURUSD, USDZAR, XAUUSD (January 2026)
# =============================================================================
cat("\n=== Part 2: Tick Data — MS-GARCH ===\n")

# Helper: load Dukascopy bid/ask parquet, compute mid, return data.frame
load_dukascopy_mid <- function(symbol, month = "202601") {
  bid_file <- file.path(data_dir, paste0(symbol, "_dukascopy_bid_", month, ".parquet"))
  ask_file <- file.path(data_dir, paste0(symbol, "_dukascopy_ask_", month, ".parquet"))

  if (!file.exists(bid_file) || !file.exists(ask_file)) {
    cat("  WARNING: parquet files not found for", symbol, "\n")
    return(NULL)
  }

  bid <- read_parquet(bid_file)
  ask <- read_parquet(ask_file)

  # Both have columns: datetime, price
  # Merge on nearest timestamp
  bid$bid <- bid$price
  ask$ask <- ask$price

  # Simple approach: use bid timestamps, merge_asof-style
  # Since both are sorted by datetime, we just align by index
  n_min <- min(nrow(bid), nrow(ask))
  df <- data.frame(
    datetime = bid$datetime[1:n_min],
    mid      = (bid$bid[1:n_min] + ask$ask[1:n_min]) / 2
  )
  return(df)
}

# Helper: load HistData LAST parquet
load_last <- function(symbol, month = "202601") {
  file <- file.path(data_dir, paste0(symbol, "_last_", month, ".parquet"))
  if (!file.exists(file)) {
    cat("  WARNING: parquet file not found for", symbol, "\n")
    return(NULL)
  }
  df <- read_parquet(file)
  df$mid <- df$price
  return(df[, c("datetime", "mid")])
}

# Fit MS-GARCH on one pair
fit_pair <- function(pair_name, df, max_pts = 5000) {
  cat("  ", pair_name, ": ", nrow(df), " raw ticks\n", sep = "")

  mid <- df$mid
  log_ret <- diff(log(mid)) * 1e4  # bps

  # Remove non-finite values
  good <- is.finite(log_ret)
  log_ret <- log_ret[good]
  times   <- df$datetime[-1][good]

  # Subsample for tractable MLE
  if (length(log_ret) > max_pts) {
    step <- floor(length(log_ret) / max_pts)
    idx  <- seq(1, length(log_ret), by = step)
    log_ret <- log_ret[idx]
    times   <- times[idx]
  }

  cat("  ", pair_name, ": fitting MS-GARCH on ", length(log_ret), " points...\n", sep = "")

  result <- tryCatch({
    spec <- CreateSpec(
      variance.spec    = list(model = c("sGARCH", "sGARCH")),
      distribution.spec = list(distribution = c("norm", "norm")),
      switch.spec      = list(do.mix = FALSE)
    )
    fit <- FitML(spec = spec, data = log_ret)

    cat("  Summary for ", pair_name, ":\n", sep = "")
    print(summary(fit))

    cond_vol <- as.numeric(Volatility(fit))

    # Save to CSV for Python/Plotly plotting
    out_df <- data.frame(
      datetime = times,
      log_ret  = log_ret,
      cond_vol = cond_vol
    )
    pair_lc <- tolower(pair_name)
    csv_path <- file.path(models_dir, paste0("ms_garch_", pair_lc, "_202601.csv"))
    write.csv(out_df, csv_path, row.names = FALSE)
    cat("  saved ", basename(csv_path), "\n", sep = "")

    # Quick R plot (PDF)
    pdf_path <- file.path(models_dir, paste0("ms_garch_", pair_lc, "_202601.pdf"))
    pdf(pdf_path, width = 10, height = 6)
    par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))
    plot(times, log_ret, type = "l", col = "steelblue", lwd = 0.5,
         main = paste(pair_name, "— Log-returns (bps)"),
         ylab = "bps", xlab = "")
    plot(times[1:length(cond_vol)], cond_vol, type = "l",
         col = "darkorange", lwd = 1,
         main = paste(pair_name, "— MS-GARCH Conditional Volatility"),
         ylab = "Volatility", xlab = "")
    dev.off()
    cat("  saved ", basename(pdf_path), "\n", sep = "")

    return(TRUE)
  }, error = function(e) {
    cat("  ", pair_name, ": MS-GARCH fitting failed — ", conditionMessage(e), "\n", sep = "")
    return(FALSE)
  })

  return(result)
}

# --- Load and fit each pair ---
# EURUSD (Dukascopy bid/ask -> mid)
df_eurusd <- load_dukascopy_mid("eurusd")
if (!is.null(df_eurusd)) fit_pair("EURUSD", df_eurusd)

# USDZAR (Dukascopy bid/ask -> mid)
df_usdzar <- load_dukascopy_mid("usdzar")
if (!is.null(df_usdzar)) fit_pair("USDZAR", df_usdzar)

# XAUUSD (HistData LAST -> mid = last)
df_xauusd <- load_last("xauusd")
if (!is.null(df_xauusd)) fit_pair("XAUUSD", df_xauusd)

cat("\nDone! Outputs saved to:\n")
cat("  PDF: ", models_dir, "\n")
cat("  CSV: ", models_dir, " (for Python/Plotly plotting)\n")
