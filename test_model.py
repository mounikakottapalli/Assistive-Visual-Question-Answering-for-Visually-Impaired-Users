import os
import pandas as pd
from tqdm import tqdm
from hybrid import hybrid_answer

# Path to test CSV file
TEST_CSV_PATH = "dataset/test.csv"  # Update if needed
IMAGE_FOLDER = "path_to_test_images"  # <-- Set this to your image directory

# Load the test data
test_df = pd.read_csv(TEST_CSV_PATH)

# Optional: limit for quick testing
test_df = test_df.head(100)  # comment this line for full test

# Store predictions
results = []

print("Running hybrid VQA model on test set...")

for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    image_path = os.path.join(IMAGE_FOLDER, row['image'])
    question = row['question']
    ground_truth = row['answer']
    try:
        predicted = hybrid_answer(image_path, question)
    except Exception as e:
        predicted = "error"
        print(f"Error on {row['image']}: {e}")

    results.append({
        "image": row['image'],
        "question": question,
        "ground_truth": ground_truth,
        "prediction": predicted
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("test_results.csv", index=False)

print("✅ Done. Results saved to test_results.csv")
