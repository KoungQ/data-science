import numpy as np
import pandas as pd

# Step 2-1: Create categorical variables
colors = np.array(["Red", "Green", "Blue", "Green", "Red", "Blue"])
sizes = np.array(["Small", "Medium", "Large", "Small", "Large", "Medium"])
brands = np.array(["Nike", "Adidas", "Puma", "Nike", "Puma", "Adidas"])

# Step 2-2: Label Encoding for brands
# "Nike" → 0, "Adidas" → 1, "Puma" → 2 (based on alphabetical order)
unique_brands = np.unique(brands)
brand_to_label = {name: idx for idx, name in enumerate(unique_brands)}
label_encoded_brands = np.array([brand_to_label[b] for b in brands])

# Step 2-3: Ordinal Encoding for sizes
# "Small" → 1, "Medium" → 2, "Large" → 3 (manual ranking)
size_order = {"Small": 1, "Medium": 2, "Large": 3}
ordinal_encoded_sizes = np.array([size_order[s] for s in sizes])

# Step 2-4: One-Hot Encoding for colors
unique_colors = np.unique(colors)  # ['Blue', 'Green', 'Red']
one_hot_encoded_colors = np.zeros((len(colors), len(unique_colors)))
color_to_index = {color: idx for idx, color in enumerate(unique_colors)}
for i, color in enumerate(colors):
    one_hot_encoded_colors[i, color_to_index[color]] = 1

# Step 2-5: Combine all features into a final matrix (6x5)
final_feature_matrix = np.hstack([
    one_hot_encoded_colors,
    ordinal_encoded_sizes.reshape(-1, 1),
    label_encoded_brands.reshape(-1, 1)
])

# Display as DataFrame
columns = [f"Color_{c}" for c in unique_colors] + ["Size_Ordinal", "Brand_Label"]
df_encoded = pd.DataFrame(final_feature_matrix, columns=columns, dtype=int)
print(df_encoded)

# Step 2-6: Explanation of encoding choices
"""
Encoding Method Explanations

1. Label Encoding (brands)
- "Nike" → 0, "Adidas" → 1, "Puma" → 2
- Efficient and compact, but can imply unintended order.
- Suitable for tree-based models, not linear models.

2. Ordinal Encoding (sizes)
- "Small" → 1, "Medium" → 2, "Large" → 3
- Appropriate because sizes have a natural ranking.

3. One-Hot Encoding (colors)
- Colors are nominal with no order.
- One-hot avoids falsely implying numerical relationships.
- Each color is treated independently in models.
"""
