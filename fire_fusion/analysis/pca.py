import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..dataset.build import FeatureGrid


def extract_features_from_batch(batch):
    """
    Extract the feature tensor from a batch.
    Adjust this function if your batch format is different.
    """
    # Case 1: batch is (features, labels) or (features, labels, meta)
    if isinstance(batch, (list, tuple)):
        features_tensor = batch[0]
        return features_tensor

    # Case 2: batch is a dict with 'x' or 'features' keys
    if isinstance(batch, dict):
        if "x" in batch:
            return batch["x"]
        if "features" in batch:
            return batch["features"]

    raise ValueError(
        f"Unrecognized batch format: {type(batch)}. "
        "Update `extract_features_from_batch` accordingly."
    )


def build_feature_matrix_from_loaders(
    loaders,
    max_samples: int = 200_000,
) -> np.ndarray:
    """
    Take a list of PyTorch DataLoaders and build a 2D feature matrix
    of shape [n_samples, n_features] for PCA.

    Parameters
    ----------
    loaders : list[DataLoader]
        For example: [fg.train_loader, fg.eval_loader]
    max_samples : int
        Upper bound on total number of samples to pull for PCA.

    Returns
    -------
    X_all : np.ndarray
        Feature matrix of shape [n_samples, n_features]
    """
    feature_list = []
    num_collected = 0

    for loader in loaders:
        for batch in loader:
            features_tensor = extract_features_from_batch(batch)

            # Move to CPU and convert to numpy
            features_np = features_tensor.detach().cpu().numpy()

            feature_list.append(features_np)

            num_collected += features_np.shape[0]
            if num_collected >= max_samples:
                break

        if num_collected >= max_samples:
            break

    X_all = np.concatenate(feature_list, axis=0)[:max_samples]
    return X_all


if __name__ == "__main__":
    fg = FeatureGrid(
        mode="load",
        load_datasets=["train", "eval"],
        start_date="2000-01-01",
        end_date="2005-12-31",
    )

    loaders = [fg.train_loader, fg.eval_loader]  # or [fg.train_loader, fg.test_loader]
    X = build_feature_matrix_from_loaders(
        loaders=loaders,
        max_samples=200_000,  # adjust if you want more/less
    )

    print("Feature matrix shape:", X.shape)  # (n_samples, n_features)

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Fit PCA
    pca = PCA(n_components=None, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print("Explained variance ratio (first 10 PCs):", explained_var[:10])
    print("Cumulative explained variance (first 10 PCs):", cumulative_var[:10])

    try:
        feature_names = fg.feature_names
    except AttributeError:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    loadings = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(pca.n_components_)],
    )

    # Show top contributing features for the first few PCs
    num_pcs_to_inspect = 5
    for pc in range(num_pcs_to_inspect):
        pc_name = f"PC{pc+1}"
        print(f"\nTop loadings for {pc_name}:")
        print(
            loadings.loc[pc_name].abs() # .sort_values(ascending=False)
            .head(10)
        )