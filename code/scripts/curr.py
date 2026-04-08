# ==========================================
# 11. DYNAMIC & STABLE PAIRS REGIME MODEL
# ==========================================
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. CONFIGURATION & AUTO-NAMING
# ==========================================
DATA_DIR = Path('../data/processed/')

# Just change these two filenames to whatever you want to test next!
FILE_A = 'eurnok_dukascopy_ask_202601.parquet'
FILE_B = 'eursek_dukascopy_ask_202601.parquet'

def extract_symbol_name(filename):
    """Extracts 'AUD/USD' from 'audusd_dukascopy_ask_202601.parquet'"""
    base = filename.split('_')[0].upper()
    if len(base) == 6:
        return f"{base[:3]}/{base[3:]}"
    return base

# Auto-generate formatting names
NAME_A = extract_symbol_name(FILE_A)
NAME_B = extract_symbol_name(FILE_B)

print(f"--- Initializing Pairs Engine: {NAME_A} vs {NAME_B} ---")

# ==========================================
# 1. LOAD AND ALIGN TICK DATA
# ==========================================
df_a = pd.read_parquet(DATA_DIR / FILE_A)
df_b = pd.read_parquet(DATA_DIR / FILE_B)

print(f"{NAME_A} raw ticks: {len(df_a):,}")
print(f"{NAME_B} raw ticks: {len(df_b):,}")

# ==========================================
# 1b. RESAMPLE TO COMMON FREQUENCY
# ==========================================
df_a = df_a.set_index('datetime').sort_index()
df_b = df_b.set_index('datetime').sort_index()

a_1m = df_a['price'].resample('1min').last()
b_1m = df_b['price'].resample('1min').last()

pair_df = pd.DataFrame({
    'Asset_A': a_1m, 
    'Asset_B': b_1m, 
}).dropna()

# Remove stale quotes
a_changed = pair_df['Asset_A'].diff().ne(0)
b_changed = pair_df['Asset_B'].diff().ne(0)
either_moved = a_changed | b_changed
pair_df = pair_df[either_moved | (pair_df.index == pair_df.index[0])]

time_gaps = pair_df.index.to_series().diff()
large_gaps = time_gaps[time_gaps > pd.Timedelta(minutes=5)]
print(f"\nAligned active-market bars: {len(pair_df):,}")
print(f"Gaps > 5 min (weekends/sessions): {len(large_gaps)}")

if len(pair_df) < 100:
    raise ValueError(f"Only {len(pair_df)} aligned bars — check data overlap / timezone handling.")

# ==========================================
# 2. Z-SCORE STANDARDIZATION
# ==========================================
raw_log_spread = np.log(pair_df['Asset_A']) - np.log(pair_df['Asset_B'])

spread_mean = raw_log_spread.mean()
spread_std = raw_log_spread.std()

if spread_std < 1e-10:
    raise ValueError(f"Spread std is ~0 ({spread_std}). Assets may be identical.")

pair_df['Spread_Standardized'] = (raw_log_spread - spread_mean) / spread_std

print(f"\nStandardization Complete.")
print(f"  Raw spread   — Mean: {spread_mean:.6f}, Std: {spread_std:.6f}")
print(f"  Standardized — Mean: {pair_df['Spread_Standardized'].mean():.4f}, Std: {pair_df['Spread_Standardized'].std():.4f}")

# ==========================================
# 3. FIT MARKOV AUTOREGRESSION
# ==========================================
print(f"\nFitting Markov Autoregression for {NAME_A} vs {NAME_B}...")

spread_values = pair_df['Spread_Standardized'].dropna().values

model = sm.tsa.MarkovAutoregression(
    spread_values,
    k_regimes=2,
    order=1,
    trend='c',
    switching_trend=True,
    switching_variance=False,
)

model.initialize_known([0.5, 0.5])

# THE SURGICAL FIX (Required for stubborn pairs like EURSEK/EURNOK)
initial_guess = model.start_params.copy()

for i, name in enumerate(model.param_names):
    if 'const' in name:
        if 'Regime 0' in name or '[0]' in name:
            initial_guess[i] = -1.0 # Force Regime 0 down
        else:
            initial_guess[i] = 1.0  # Force Regime 1 up
    elif 'sigma2' in name:
        initial_guess[i] = 0.1      # Prevent divide-by-zero crashes
    elif 'ar.L1' in name:
        initial_guess[i] = 0.95     # Assume high momentum
    elif 'p[0->0]' in name:
        initial_guess[i] = 0.99     # Assume sticky regimes
    elif 'p[1->0]' in name:
        initial_guess[i] = 0.01

try:
    # Notice we are passing start_params=initial_guess here!
    result = model.fit(start_params=initial_guess, em_iter=0, method='bfgs', disp=False)
except Exception as e:
    print(f"bfgs failed ({e}), falling back to powell...")
    result = model.fit(start_params=initial_guess, em_iter=0, method='powell', disp=False)

print(result.summary())

# ==========================================
# 4. EXTRACT REGIME MEANS SAFELY
# ==========================================
param_names = model.param_names
params = np.array(result.params) 

const_indices = [i for i, n in enumerate(param_names) if 'const' in n.lower()]

if len(const_indices) >= 2:
    c0 = params[const_indices[0]]
    c1 = params[const_indices[1]]
else:
    c0 = params[0]
    c1 = params[1]

print(f"\nRegime 0 mean (standardized): {c0:.4f}")
print(f"Regime 1 mean (standardized): {c1:.4f}")
print(f"Regime 0 mean (original log-spread): {c0 * spread_std + spread_mean:.6f}")
print(f"Regime 1 mean (original log-spread): {c1 * spread_std + spread_mean:.6f}")

# ==========================================
# 5. VISUALIZE (sequential x-axis)
# ==========================================
fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

prob_state_1 = result.smoothed_marginal_probabilities[:, 1]
n_obs = len(prob_state_1)
plot_df = pair_df.dropna(subset=['Spread_Standardized']).iloc[-n_obs:]
plot_spread = spread_values[-n_obs:]

x = np.arange(n_obs)

dates = plot_df.index.date
day_changes = np.where(dates[:-1] != dates[1:])[0] + 1
label_every = max(1, len(day_changes) // 15)
tick_positions = day_changes[::label_every]
tick_labels = [str(dates[i]) for i in tick_positions]

axs[0].plot(x, plot_spread, color='indigo', linewidth=0.5, label='Standardized Spread')
axs[0].axhline(c0, color='red', linestyle='--', label=f'Regime 0 Mean: {c0:.2f}')
axs[0].axhline(c1, color='green', linestyle='--', label=f'Regime 1 Mean: {c1:.2f}')
axs[0].set_title(f'{NAME_A} vs {NAME_B} — Standardized Log-Spread Regime Analysis')
axs[0].set_ylabel('Z-Score')
axs[0].legend()

for pos in day_changes:
    gap = plot_df.index[pos] - plot_df.index[pos - 1]
    if gap > pd.Timedelta(hours=6):
        for ax in axs:
            ax.axvline(pos, color='gray', alpha=0.3, linewidth=0.5, linestyle=':')

axs[1].fill_between(x, 0, prob_state_1, color='seagreen', alpha=0.5)
axs[1].set_title('Smoothed Probability of Regime 1')
axs[1].set_ylabel('P(Regime 1)')
axs[1].set_ylim(-0.05, 1.05)

axs[1].set_xticks(tick_positions)
axs[1].set_xticklabels(tick_labels, rotation=45, ha='right')

plt.tight_layout()

# Dynamic save name (e.g., regime_analysis_AUDUSD_NZDUSD.png)
save_name = f'regime_analysis_{NAME_A.replace("/", "")}_{NAME_B.replace("/", "")}.png'
plt.savefig(save_name, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {save_name}")