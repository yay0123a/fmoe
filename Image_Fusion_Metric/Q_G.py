import numpy as np
from scipy.ndimage import convolve

import cv2
import os
"""
python 实现论文：Objective assessment of multiresolution image fusion algorithms for context enhancement in night vision: A comparative study
指标：Gradient-BasedFusionPerformance (Q_G)

"""

def metric_xydeas(img1, img2, fuse):
    """
    Python implementation of Xydeas fusion metric

    Parameters:
        img1, img2 -- input images (2D numpy arrays)
        fuse       -- fused image (2D numpy array)

    Returns:
        res -- metric value

    Reference:
        Xydeas, C. S., & Petrovic, V. (2000).
        Objective image fusion performance measure.
        Electronics Letters, 36(4), 308-309.
    """

    # Sobel filters
    flt1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # horizontal
    flt2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # vertical

    # 1) Get edge maps
    def compute_gradient_components(img):
        imgX = convolve(img, flt1, mode='reflect')
        imgY = convolve(img, flt2, mode='reflect')
        imgG = np.sqrt(imgX ** 2 + imgY ** 2)

        # Avoid division by zero
        buffer = (imgX == 0).astype(float) * 0.00001
        imgX = imgX + buffer
        imgA = np.arctan(imgY / imgX)

        return imgX, imgY, imgG, imgA

    fuseX, fuseY, fuseG, fuseA = compute_gradient_components(fuse)
    img1X, img1Y, img1G, img1A = compute_gradient_components(img1)
    img2X, img2Y, img2G, img2A = compute_gradient_components(img2)

    # 2) Edge preservation estimation
    def compute_q(G, A, imgG, imgA, fuseG, fuseA):
        bimap = imgG > fuseG

        # Avoid division by zero
        buffer = (imgG == 0).astype(float) * 0.00001
        imgG = imgG + buffer
        buffer1 = fuseG / imgG

        buffer = (fuseG == 0).astype(float) * 0.00001
        fuseG = fuseG + buffer
        buffer2 = imgG / fuseG

        G_af = bimap * buffer1 + (1 - bimap) * buffer2
        A_af = 1 - np.abs(imgA - fuseA) * 2 / np.pi

        # Parameters (modified according to Fig. 1)
        gamma1 = 0.9994
        gamma2 = 0.9879
        k1 = -15
        k2 = -22
        delta1 = 0.5
        delta2 = 0.8

        Qg_AF = gamma1 / (1 + np.exp(k1 * (G_af - delta1)))
        Qalpha_AF = gamma2 / (1 + np.exp(k2 * (A_af - delta2)))
        Q_af = Qg_AF * Qalpha_AF

        return Q_af

    Qaf = compute_q(img1G, img1A, img1G, img1A, fuseG, fuseA)
    Qbf = compute_q(img2G, img2A, img2G, img2A, fuseG, fuseA)

    # 3) Compute weighting matrix
    L = 1
    Wa = img1G ** L
    Wb = img2G ** L

    numerator = np.sum(Qaf * Wa + Qbf * Wb)
    denominator = np.sum(Wa + Wb)
    res = numerator / denominator

    return res

if __name__ == '__main__':
    """
    有一点点误差，但是几乎和matlab的结果一致
    """
    # Load images (convert to grayscale if they're color)
    root_path = './data/TNO'
    ir_path = os.path.join(root_path, 'ir')
    vi_path = os.path.join(root_path, 'vi')
    fus_U2Fusion_path = os.path.join(root_path, 'fus_U2Fusion')

    score_list = []

    for root, dirs, files in os.walk(ir_path):
        for file in files:
            img1 = cv2.imread(os.path.join(ir_path, file), 0).astype(float)
            img2 = cv2.imread(os.path.join(vi_path, file), 0).astype(float)
            fused = cv2.imread(os.path.join(fus_U2Fusion_path, file), 0).astype(float)
            # Compute metric
            score = metric_xydeas(img1, img2, fused)
            print(f"{file}>>>Fusion quality score: {score}")
            score_list.append(score)
    score_avg = np.mean(score_list)
    print(f"Fusion quality AVG score: {score_avg}")

    # img1 = cv2.imread('./data/TNO/ir/01.png', 0).astype(float)
    # img2 = cv2.imread('./data/TNO/vi/01.png', 0).astype(float)
    # fused = cv2.imread('./data/TNO/fus_U2Fusion/01.png', 0).astype(float)

    # Compute metric
    # score = metric_xydeas(img1, img2, fused)
    # print(f"Fusion quality score: {score}")