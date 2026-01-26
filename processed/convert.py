import joblib
import os

# 1. Name of your current large file
input_file = 'y_train_ready.joblib' 
# 2. Name of the new compressed file
output_file = 'y_train_ready_compressed.joblib'

print(f"Loading {input_file}...")
# Load the uncompressed data
data = joblib.load(input_file)

print("Compressing and saving...")
# Save with compression level 3 (good balance) or 9 (maximum)
joblib.dump(data, output_file, compress=3)

# Calculate savings
old_size = os.path.getsize(input_file) / (1024 * 1024)
new_size = os.path.getsize(output_file) / (1024 * 1024)

print(f"Original Size: {old_size:.2f} MB")
print(f"New Size: {new_size:.2f} MB")
print(f"Reduction: {((old_size - new_size) / old_size) * 100:.1f}%")