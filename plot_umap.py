import numpy as np
import umap
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='photo')
parser.add_argument('--backbone', type=str, default='gcn')
args = parser.parse_args()

# Load saved embeddings
normal_emb = np.load(f'embeddings/{args.dataset}_{args.backbone}_normal.npy')
outlier_emb = np.load(f'embeddings/{args.dataset}_{args.backbone}_outlier.npy')
anomaly_emb = np.load(f'embeddings/{args.dataset}_{args.backbone}_anomaly.npy')

print(f"Loaded: {normal_emb.shape[0]} normals, {outlier_emb.shape[0]} outliers, {anomaly_emb.shape[0]} real anomalies")

# Combine all embeddings for UMAP
all_emb = np.vstack([normal_emb, outlier_emb, anomaly_emb])

# Create labels for coloring
labels = np.array(
    ['Normal'] * len(normal_emb) + 
    ['Generated Outlier'] * len(outlier_emb) + 
    ['Real Anomaly'] * len(anomaly_emb)
)

# Run UMAP
print("Running UMAP dimensionality reduction...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(all_emb)
print("UMAP complete!")

# Split back into groups
n_normal = len(normal_emb)
n_outlier = len(outlier_emb)
n_anomaly = len(anomaly_emb)

normal_2d = embedding_2d[:n_normal]
outlier_2d = embedding_2d[n_normal:n_normal + n_outlier]
anomaly_2d = embedding_2d[n_normal + n_outlier:]

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.scatter(normal_2d[:, 0], normal_2d[:, 1], 
           c='#2E86AB', s=15, alpha=0.4, label=f'Normal ({n_normal})')
ax.scatter(anomaly_2d[:, 0], anomaly_2d[:, 1], 
           c='#EF4444', s=25, alpha=0.7, label=f'Real Anomaly ({n_anomaly})')
ax.scatter(outlier_2d[:, 0], outlier_2d[:, 1], 
           c='#F59E0B', s=25, alpha=0.7, label=f'Generated Outlier ({n_outlier})')

ax.set_title(f'UMAP Visualization - {args.dataset.capitalize()} ({args.backbone.upper()})', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.set_xlabel('UMAP 1', fontsize=12)
ax.set_ylabel('UMAP 2', fontsize=12)

plt.tight_layout()

# Save plot
os.makedirs('plots', exist_ok=True)
save_path = f'plots/umap_{args.dataset}_{args.backbone}.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"Plot saved to {save_path}")
plt.show()