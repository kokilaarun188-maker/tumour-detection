import os
import sys
import subprocess


# ==========================================
# GET PROJECT FOLDER
# ==========================================

base_dir = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# FUNCTION TO RUN EACH STEP
# ==========================================

def run_step(step_number, title, filename):

    print("\n")
    print("=" * 60)
    print(f"STEP {step_number}: {title}")
    print("=" * 60)

    file_path = os.path.join(base_dir, filename)

    if not os.path.exists(file_path):
        print(f"ERROR: {filename} not found!")
        return False

    result = subprocess.run(
        [sys.executable, file_path],
        cwd=base_dir
    )

    if result.returncode != 0:
        print("\n")
        print("=" * 60)
        print(f"STEP {step_number} FAILED")
        print(f"File: {filename}")
        print("=" * 60)

        return False

    print("\n")
    print(f"STEP {step_number} COMPLETED SUCCESSFULLY")

    return True


# ==========================================
# STEP 1 - EDA
# ==========================================

if not run_step(
    1,
    "Dataset Distribution / EDA",
    "dataset_distribution.py"
):
    print("\nProject stopped because Step 1 failed.")
    sys.exit(1)


# ==========================================
# STEP 2 - CNN TRAINING
# ==========================================

if not run_step(
    2,
    "CNN Model Training",
    "brain_tumor.py"
):
    print("\nProject stopped because Step 2 failed.")
    sys.exit(1)


# ==========================================
# STEP 3 - PREDICTION
# ==========================================

if not run_step(
    3,
    "Brain Tumor Prediction",
    "predict.py"
):
    print("\nProject stopped because Step 3 failed.")
    sys.exit(1)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n")
print("=" * 60)
print("       PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nWorkflow:")
print("1. Dataset EDA          ✓")
print("2. CNN Training         ✓")
print("3. Model Prediction     ✓")
print("\nSaved model:")
print("brain_tumor_model.h5")