import os
from collections import Counter
import matplotlib.pyplot as plt

# Set dataset path
dataset_path = "Data/speech_commands_v0.02"

# Count samples per label
class_counts = Counter()
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path) or label == "_background_noise_":
        continue
    wav_files = [f for f in os.listdir(label_path) if f.endswith(".wav")]
    class_counts[label] = len(wav_files)

# Sort by count (optional but makes the plot clearer)
sorted_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

# Plotting
plt.figure(figsize=(14, 6))
plt.bar(sorted_counts.keys(), sorted_counts.values(), color="mediumseagreen")
plt.title("Class Distribution: Full Speech Commands Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=90)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# Print exact values
for label, count in sorted_counts.items():
    print(f"{label}: {count}")
