import os

import imagehash
import numpy as np
import pandas as pd
from PIL import Image


class DeepLearningAnalyzer:
    def __init__(self):
        # Lazy loading of heavy frameworks
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        self.tf = tf

        # We load model globally to avoid reloading for every image
        self.model = None

    def initialize_model(self):
        if self.model is None:
            self.model = self.tf.keras.applications.ResNet50(
                weights="imagenet", include_top=False, pooling="avg"
            )

    def extract_embedding(self, img_rgb):
        """Extracts ResNet50 Embedding."""
        import cv2

        img_resized = cv2.resize(img_rgb, (224, 224))
        img_preprocessed = self.tf.keras.applications.resnet50.preprocess_input(
            img_resized.astype(np.float32)
        )
        emb = self.model.predict(np.expand_dims(img_preprocessed, axis=0), verbose=0)[0]
        return emb

    @staticmethod
    def get_phash(img_rgb):
        pil_img = Image.fromarray(img_rgb)
        return str(imagehash.phash(pil_img))

    @staticmethod
    def apply_dimensionality_reduction(
        df: pd.DataFrame, embeddings: list
    ) -> pd.DataFrame:
        """Applies PCA, t-SNE and IsolationForest on the embeddings."""
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest
        from sklearn.manifold import TSNE

        iso = IsolationForest(contamination=0.05, random_state=42)
        df["Is_Outlier"] = iso.fit_predict(embeddings)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(embeddings)
        df["PCA1"] = pca_result[:, 0]
        df["PCA2"] = pca_result[:, 1]

        perplexity = min(30, max(5, len(embeddings) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)
        tsne_result = tsne.fit_transform(np.array(embeddings))
        df["TSNE1"] = tsne_result[:, 0]
        df["TSNE2"] = tsne_result[:, 1]

        dups = df.duplicated(subset=["pHash"], keep=False)
        df["Is_Duplicate"] = dups

        return df
