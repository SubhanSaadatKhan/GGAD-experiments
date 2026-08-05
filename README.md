# GGAD Experiments
Experiments and extensions on GGAD (NeurIPS 2024) for the seminar Advanced ML for Anomaly Detection, FAU Erlangen-Nuremberg, SS 2026.

Paper: https://arxiv.org/abs/2402.11887
Original code: https://github.com/mala-lab/GGAD

## Experiments
- Reproducibility on Reddit, Amazon, Photo datasets
- Labeled normal ratio sensitivity (5% to 50%)
- Outlier generation ratio sensitivity (1% to 30%)
- GAT backbone extension replacing GCN
- UMAP visualizations of node embeddings

## Setup
1. Clone this repo
2. Download datasets from the original GGAD repo's Google Drive link
3. Place .mat files in a dataset/ folder
4. Install dependencies: pip install torch numpy scipy scikit-learn matplotlib networkx pandas tqdm umap-learn

## Usage
python run.py --dataset reddit --rate 0.5 --outlier_ratio 0.15 --backbone gcn
python run.py --dataset photo --backbone gat
python plot_umap.py --dataset reddit --backbone gcn

## Author
Subhan Saadat - M.Sc. Data Science, FAU Erlangen-Nuremberg
