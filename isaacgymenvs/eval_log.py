import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns

# File path
log_file = "eval_log.txt"

# Lists to store data
direct_successes = []
post_reset_successes = []

# Regex patterns
direct_pattern = r"Direct average consecutive successes = ([\d\.]+)"
post_reset_pattern = r"Post-Reset average consecutive successes = ([\d\.]+)"

# Read file
with open(log_file, "r") as f:
    for line in f:
        direct_match = re.search(direct_pattern, line)
        if direct_match:
            direct_successes.append(float(direct_match.group(1)))

        post_reset_match = re.search(post_reset_pattern, line)
        if post_reset_match:
            post_reset_successes.append(float(post_reset_match.group(1)))

# Ensure both lists are the same length
min_length = min(len(direct_successes), len(post_reset_successes))
direct_successes = direct_successes[:min_length]
post_reset_successes = post_reset_successes[:min_length]

# Limit to first 5000 points
limit = 5000
direct_successes = direct_successes[:limit]
post_reset_successes = post_reset_successes[:limit]

# Convert to DataFrame for easy smoothing
df = pd.DataFrame({
    'Direct': direct_successes,
    'Post-Reset': post_reset_successes
})

# Apply Rolling Mean (Smoothing)
window_size = 50  # Adjust this to control smoothness (higher = smoother)
df_smooth = df.rolling(window=window_size).mean()

# Set Style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# Plot Raw Data (faint background)
plt.plot(df['Direct'], color='blue', alpha=0.15, linewidth=1, label='_nolegend_')
plt.plot(df['Post-Reset'], color='orange', alpha=0.15, linewidth=1, label='_nolegend_')

# Plot Smoothed Data (solid lines)
plt.plot(df_smooth['Direct'], label=f"Direct Average (Moving Avg {window_size})", color='royalblue', linewidth=2.5)
plt.plot(df_smooth['Post-Reset'], label=f"Post-Reset Average (Moving Avg {window_size})", color='darkorange', linewidth=2.5)

# Formatting
plt.title(f"Fix Egg Size, Lemon(Fixed Size)(First {len(direct_successes)} Steps)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Step / Update", fontsize=14)
plt.ylabel("Consecutive Successes", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
plt.xlim(0, limit)
plt.ylim(bottom=0)  # Ensure y-axis starts at 0

# Save and Show
plt.tight_layout()
plt.savefig("evaluation_plot_improved.png", dpi=300)
plt.show()

print(f"Improved plot saved to evaluation_plot_improved.png")

# plt.title(f"Evaluation Success Rate of No Randomization Rubic Cube(First {len(direct_successes)} Steps)")