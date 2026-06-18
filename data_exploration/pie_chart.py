import matplotlib.pyplot as plt
import numpy as np

# Set style for better looking charts
plt.style.use('seaborn-v0_8')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ------------------- Chart 1: Credits Progress -------------------
# Data
credits_obtained = 33
credits_total = 60
credits_remaining = credits_total - credits_obtained

# Create pie chart
sizes = [credits_obtained, credits_remaining]
labels = [f'Obtained\n{credits_obtained} credits', f'Remaining\n{credits_remaining} credits']
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)  # Slightly explode the obtained slice

ax1.pie(sizes, labels=labels, colors=colors, explode=explode, 
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title(f'Training Progress: {credits_obtained}/{credits_total} Credits', 
              fontsize=14, weight='bold', pad=20)

# Add a circle in the middle to create a donut effect
centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=1, edgecolor='#333')
ax1.add_artist(centre_circle)

# ------------------- Chart 2: Training Type Distribution -------------------
# Data
training_types = ['DF1', 'DF2', 'DF3']
counts = [15, 14, 19]
colors2 = ['#3498db', '#f39c12', '#9b59b6']

# Create pie chart
wedges, texts, autotexts = ax2.pie(counts, labels=training_types, colors=colors2,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 12, 'weight': 'bold'},
                                    explode=(0.02, 0.02, 0.02))

# Customize the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
    autotext.set_weight('bold')

ax2.set_title('Training Type Distribution', fontsize=14, weight='bold', pad=20)

# Add a legend
ax2.legend(wedges, [f'{typ} ({cnt} trainings)' for typ, cnt in zip(training_types, counts)],
           title='Training Types', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the charts
plt.show()

# Optionally save the figure
# plt.savefig('training_progress_charts.png', dpi=300, bbox_inches='tight')