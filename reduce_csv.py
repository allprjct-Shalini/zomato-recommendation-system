import pandas as pd

print("Loading original large dataset...")
# Load your large dataset
df = pd.read_csv("dataset/zomato.csv")

print(f"Original shape: {df.shape}")

# Take a random sample of 10,000 rows (this will be around 5-10 MB, well under GitHub's limit)
df_small = df.sample(n=10000, random_state=42)

print(f"New sampled shape: {df_small.shape}")

# Save it as a new small CSV file
df_small.to_csv("dataset/zomato_small.csv", index=False)
print("Successfully created dataset/zomato_small.csv!")