import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

print("Loading model...")

model = load_model("brain_tumor_model.h5")

print("Model loaded successfully!")


# ==========================================
# 2. CLASS NAMES
# ==========================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

test_folder = "Testing"


# ==========================================
# 3. VARIABLES
# ==========================================

total_probs = np.zeros(4)
total_images = 0


# ==========================================
# 4. PROCESS ALL TESTING IMAGES
# ==========================================

for folder in class_names:

    folder_path = os.path.join(test_folder, folder)

    print(f"\nChecking folder: {folder_path}")

    if not os.path.exists(folder_path):
        print("Folder not found!")
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    print(f"Found {len(files)} images")

    for file in files:

        img_path = os.path.join(
            folder_path,
            file
        )

        # Load image
        img = image.load_img(
            img_path,
            target_size=(128, 128)
        )

        # Convert image to array
        img = image.img_to_array(img)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        predictions = model.predict(
            img,
            verbose=0
        )

        # Add probabilities
        total_probs += predictions[0]

        total_images += 1


# ==========================================
# 5. DISPLAY TOTAL IMAGES
# ==========================================

print("\n======================================")
print(f"Total Images Processed: {total_images}")
print("======================================")


# ==========================================
# 6. CALCULATE AVERAGE PROBABILITY
# ==========================================

if total_images > 0:

    average_probs = (
        total_probs / total_images
    ) * 100


    # ======================================
    # 7. DISPLAY RESULTS
    # ======================================

    print("\n===== OVERALL PREDICTION PERCENTAGE =====")

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name:<15}: "
            f"{average_probs[i]:.2f}%"
        )


    # ======================================
    # 8. RESULTS TABLE
    # ======================================

    print("\n======================================")
    print("          PREDICTION TABLE")
    print("======================================")

    print(
        f"{'Tumor Class':<20}"
        f"{'Probability'}"
    )

    print("--------------------------------------")

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name:<20}"
            f"{average_probs[i]:.2f}%"
        )

    print("======================================")


    # ======================================
    # 9. FIND HIGHEST PROBABILITY CLASS
    # ======================================

    predicted_index = np.argmax(average_probs)

    predicted_class = class_names[predicted_index]

    predicted_probability = average_probs[predicted_index]


    print("\n===== FINAL RESULT =====")

    print(
        f"Predicted Class : {predicted_class}"
    )

    print(
        f"Probability     : "
        f"{predicted_probability:.2f}%"
    )


    # ======================================
    # 10. BAR CHART
    # ======================================

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        class_names,
        average_probs
    )

    plt.title(
        "Overall Brain Tumor Prediction Probability"
    )

    plt.xlabel("Tumor Class")

    plt.ylabel("Average Probability (%)")

    plt.ylim(0, 100)

    # Display percentages above bars
    for bar, value in zip(
        bars,
        average_probs
    ):

        plt.text(
            bar.get_x() +
            bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.show()


    # ======================================
    # 11. PIE CHART
    # ======================================

    plt.figure(figsize=(7, 7))

    plt.pie(
        average_probs,
        labels=class_names,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Overall Brain Tumor Prediction Distribution"
    )

    plt.tight_layout()

    plt.show()


else:

    print("\nNo images were processed.")