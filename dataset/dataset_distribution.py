import os
import matplotlib.pyplot as plt

# ==============================
# Dataset Configuration
# ==============================

classes = ["glioma", "meningioma", "notumor", "pituitary"]
test_folder = "Testing"

counts = {}
total = 0

# ==============================
# Count Images
# ==============================

for cls in classes:

    folder = os.path.join(test_folder, cls)

    if not os.path.exists(folder):
        print(f"Warning: Folder not found - {folder}")
        counts[cls] = 0
        continue

    num_images = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    counts[cls] = num_images
    total += num_images


# ==============================
# Calculate Percentages
# ==============================

percentages = {}

for cls in classes:
    if total > 0:
        percentages[cls] = (counts[cls] / total) * 100
    else:
        percentages[cls] = 0


# ==============================
# Console Output
# ==============================

print("\n===================================")
print("       DATASET DISTRIBUTION")
print("===================================")

print(f"{'Class':<15} {'Images':<10} {'Percentage'}")
print("-----------------------------------")

for cls in classes:
    print(
        f"{cls:<15} "
        f"{counts[cls]:<10} "
        f"{percentages[cls]:.2f}%"
    )

print("-----------------------------------")
print(f"{'TOTAL':<15} {total:<10} 100.00%")

print("===================================\n")


# ==============================
# Bar Chart
# ==============================

plt.figure(figsize=(8, 5))

bars = plt.bar(classes, [counts[cls] for cls in classes])

plt.title("Brain Tumor Dataset Distribution")
plt.xlabel("Tumor Class")
plt.ylabel("Number of Images")

# Display values above bars
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# ==============================
# Pie Chart
# ==============================

plt.figure(figsize=(7, 7))

plt.pie(
    [counts[cls] for cls in classes],
    labels=classes,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Brain Tumor Dataset Percentage Distribution")

plt.show()


# ==============================
# Summary
# ==============================

print("EDA completed successfully!")
print(f"Total testing images analyzed: {total}")