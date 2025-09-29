# Script untuk membuat placeholder images
import matplotlib.pyplot as plt
import numpy as np

# Create missing values visualization placeholder
plt.figure(figsize=(10, 6))
np.random.seed(42)
features = ['Feature_' + str(i) for i in range(1, 11)]
missing_pct = np.random.uniform(0, 30, 10)
plt.barh(features, missing_pct)
plt.xlabel('Missing Percentage (%)')
plt.title('Missing Values Analysis')
plt.tight_layout()
plt.savefig('images/missing_values.png', dpi=300, bbox_inches='tight')
plt.close()

# Create feature importance placeholder
plt.figure(figsize=(10, 8))
importance_scores = np.random.uniform(0.01, 0.15, 15)[::-1]
feature_names = ['Feature_' + str(i) for i in range(1, 16)]
plt.barh(feature_names, importance_scores)
plt.xlabel('Importance Score')
plt.title('Feature Importance from Best Model')
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Placeholder images created successfully!")