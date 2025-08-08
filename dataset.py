from datasets import load_dataset
import pandas as pd

# Load dataset from Hugging Face
dataset = load_dataset("lmms-lab/VizWiz-VQA")
df = pd.DataFrame(dataset['train'])

# Extract only image, question, and first answer (or mark unanswerable)
df = df[['image', 'question', 'answers']]
df['answer'] = df['answers'].apply(lambda x: x[0] if x else 'unanswerable')
df = df.drop(columns=['answers'])

# Shuffle and split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = df.iloc[:10000]
remaining = df.iloc[5000:]
val_df = remaining.iloc[:int(0.5 * len(remaining))]
test_df = remaining.iloc[int(0.5 * len(remaining)):]

# Save to CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
