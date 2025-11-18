"""
Encodeur facial basé sur MTCNN + Deep Features
Plus précis que HOG/Texture pour la reconnaissance faciale
"""
import cv2
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FaceNetEncoder:
    """
    Encodeur facial utilisant des features deep learning plus discriminantes
    """

    def __init__(self, embedding_size: int = 512):
        self.embedding_size = embedding_size
        self.target_size = (160, 160)

    def preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Prétraitement minimal pour préserver les détails discriminants
        """
        # Redimensionner
        face_resized = cv2.resize(face_img, self.target_size)

        # Normalisation simple (0-1)
        face_normalized = face_resized.astype(np.float32) / 255.0

        # Standardisation (mean=0, std=1)
        mean = np.mean(face_normalized)
        std = np.std(face_normalized)
        face_standardized = (face_normalized - mean) / (std + 1e-7)

        return face_standardized

    def extract_deep_features(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extraction de features deep plus discriminantes
        """
        # Prétraitement
        face_processed = self.preprocess_face(face_img)

        # Convertir en niveaux de gris pour certaines features
        if len(face_processed.shape) == 3:
            gray = cv2.cvtColor((face_processed * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (face_processed * 255).astype(np.uint8)

        # 1. Features HOG plus fines (plus de détails)
        hog_features = self._compute_fine_hog(gray)

        # 2. Features LBP (Local Binary Patterns) - très discriminantes
        lbp_features = self._compute_lbp_features(gray)

        # 3. Features de gradient (Sobel)
        gradient_features = self._compute_gradient_features(gray)

        # 4. Features de fréquence (DCT)
        frequency_features = self._compute_frequency_features(gray)

        # 5. Features de texture avancées
        texture_features = self._compute_advanced_texture(gray)

        # Concaténation
        all_features = np.concatenate([
            hog_features,
            lbp_features,
            gradient_features,
            frequency_features,
            texture_features
        ])

        # Réduction dimensionnelle
        embedding = self._reduce_dimensions(all_features, self.embedding_size)

        # Normalisation L2
        embedding = embedding / (np.linalg.norm(embedding) + 1e-7)

        return embedding

    def _compute_fine_hog(self, gray_img: np.ndarray) -> np.ndarray:
        """HOG avec paramètres plus fins pour plus de discrimination"""
        win_size = (160, 160)
        block_size = (8, 8)  # Plus petit pour plus de détails
        block_stride = (4, 4)  # Stride plus petit
        cell_size = (4, 4)  # Cellules plus petites
        nbins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(gray_img)

        return features.flatten()

    def _compute_lbp_features(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Local Binary Patterns - Très discriminant pour les visages
        """
        h, w = gray_img.shape
        lbp = np.zeros_like(gray_img)

        # LBP sur grille 3x3
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray_img[i, j]
                code = 0
                code |= (gray_img[i-1, j-1] >= center) << 7
                code |= (gray_img[i-1, j] >= center) << 6
                code |= (gray_img[i-1, j+1] >= center) << 5
                code |= (gray_img[i, j+1] >= center) << 4
                code |= (gray_img[i+1, j+1] >= center) << 3
                code |= (gray_img[i+1, j] >= center) << 2
                code |= (gray_img[i+1, j-1] >= center) << 1
                code |= (gray_img[i, j-1] >= center) << 0
                lbp[i, j] = code

        # Histogramme LBP par région (8x8 régions)
        cell_h, cell_w = h // 8, w // 8
        hist_features = []

        for i in range(8):
            for j in range(8):
                cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist = np.histogram(cell, bins=256, range=(0, 256))[0]
                hist = hist.astype(np.float32)
                hist = hist / (hist.sum() + 1e-7)  # Normalisation
                hist_features.extend(hist)

        return np.array(hist_features)

    def _compute_gradient_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Features basées sur les gradients (Sobel)"""
        # Gradients Sobel
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude et direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)

        # Statistiques par région
        h, w = gray_img.shape
        cell_h, cell_w = h // 4, w // 4

        features = []
        for i in range(4):
            for j in range(4):
                mag_cell = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                dir_cell = direction[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                features.append(np.mean(mag_cell))
                features.append(np.std(mag_cell))
                features.append(np.mean(dir_cell))
                features.append(np.std(dir_cell))

        return np.array(features)

    def _compute_frequency_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Features de fréquence (DCT - Discrete Cosine Transform)"""
        # DCT sur l'image entière
        dct = cv2.dct(gray_img.astype(np.float32))

        # Garder les coefficients basse fréquence (plus importants)
        dct_low = dct[:40, :40]

        return dct_low.flatten()

    def _compute_advanced_texture(self, gray_img: np.ndarray) -> np.ndarray:
        """Features de texture avancées (variance, énergie, entropie)"""
        h, w = gray_img.shape
        cell_h, cell_w = h // 8, w // 8

        features = []
        for i in range(8):
            for j in range(8):
                cell = gray_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

                # Variance
                features.append(np.var(cell))

                # Énergie
                features.append(np.sum(cell**2))

                # Entropie
                hist, _ = np.histogram(cell, bins=256, range=(0, 256))
                hist = hist / (hist.sum() + 1e-7)
                entropy = -np.sum(hist * np.log2(hist + 1e-7))
                features.append(entropy)

        return np.array(features)

    def _reduce_dimensions(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Réduction de dimensions par PCA simple (moyenne par blocs)"""
        if len(features) <= target_dim:
            padding = target_dim - len(features)
            return np.pad(features, (0, padding), mode='constant')

        # Moyennage par blocs
        step = len(features) / target_dim
        reduced = []

        for i in range(target_dim):
            start = int(i * step)
            end = int((i + 1) * step)
            reduced.append(np.mean(features[start:end]))

        return np.array(reduced)

    def encode_face(self, face_img: np.ndarray) -> List[float]:
        """
        Génère l'embedding FaceNet pour un visage

        Args:
            face_img: Image du visage (BGR)

        Returns:
            Liste de floats (embedding de taille self.embedding_size)
        """
        try:
            embedding = self.extract_deep_features(face_img)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage FaceNet: {e}")
            raise


if __name__ == "__main__":
    # Test rapide
    encoder = FaceNetEncoder(embedding_size=512)

    # Créer une image test
    test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Encoder
    encoding = encoder.encode_face(test_img)

    print(f"Encodage généré: {len(encoding)} dimensions")
    print(f"Première valeur: {encoding[0]:.6f}")
    print(f"Norme L2: {np.linalg.norm(encoding):.6f}")
