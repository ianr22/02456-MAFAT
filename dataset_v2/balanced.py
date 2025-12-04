import pandas as pd

# ===== 1. Load the original CSV =====
csv_path = "train.csv"

print(f"Loading CSV from: {csv_path}")
df = pd.read_csv(csv_path)

print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

if "general_class" not in df.columns:
    raise KeyError("Column 'general_class' not found in train.csv. Check your header.")

# ===== 2. Inspect general_class values =====
print("\nUnique values in general_class:")
print(df["general_class"].unique())

# Try to automatically pick small/large labels by substring
values = df["general_class"].astype(str).unique()

small_candidates = [v for v in values if "small" in v.lower()]
large_candidates = [v for v in values if "large" in v.lower()]

print("\nDetected small candidates:", small_candidates)
print("Detected large candidates:", large_candidates)

if len(small_candidates) == 0 or len(large_candidates) == 0:
    raise ValueError("Could not find 'small' or 'large' in general_class values. "
                     "Check the unique values printed above.")

if len(small_candidates) > 1 or len(large_candidates) > 1:
    print("\nWARNING: More than one candidate for small/large. "
          "Using the first match for each.")
    
SMALL_LABEL = small_candidates[0]
LARGE_LABEL = large_candidates[0]

print(f"\nUsing SMALL_LABEL = {SMALL_LABEL!r}")
print(f"Using LARGE_LABEL = {LARGE_LABEL!r}")

# ===== 3. Split into small vs large vehicles =====
df_small = df[df["general_class"] == SMALL_LABEL]
df_large = df[df["general_class"] == LARGE_LABEL]

print(f"\nSmall vehicles: {len(df_small)}")
print(f"Large vehicles: {len(df_large)}")

if len(df_small) == 0 or len(df_large) == 0:
    raise ValueError("One of the classes has zero samples after filtering. "
                     "Check SMALL_LABEL and LARGE_LABEL.")

# ===== 4. Sample equal number from each class =====
n = min(len(df_small), len(df_large))
print(f"\nBalancing to n = {n} per class")

df_small_sample = df_small.sample(n, random_state=42)
df_large_sample = df_large.sample(n, random_state=42)

# ===== 5. Combine into a balanced subset =====
balanced_df = pd.concat([df_small_sample, df_large_sample],
                        ignore_index=True)

print("\nBalanced dataset size:", len(balanced_df))
print("Class counts in balanced set:")
print(balanced_df["general_class"].value_counts())

# ===== 6. Save to a new CSV =====
out_path = "train_balanced_general_class.csv"
balanced_df.to_csv(out_path, index=False)
print(f"\nSaved balanced subset to: {out_path}")
