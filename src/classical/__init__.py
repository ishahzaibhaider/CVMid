"""Classical computer-vision toolkit for VeriVision.

VeriVision started as a deep-learning AI-image detector (Unit 5 / Unit 7 of the
AIC341 syllabus). This package extends it into a *full-syllabus* project: one
runnable module per course unit, each producing real result figures that land
in ``reports/figures/classical/`` and are embedded in the written report.

Unit map
--------
* ``unit1_image_basics``   — image formation, the CV pipeline, channels & colour spaces.
* ``unit2_sampling``       — sampling & aliasing, up/down-sampling, image pyramids.
* ``unit2_features``       — edges, corners, blobs, and ORB keypoints/descriptors.
* ``unit2_hough_ransac``   — Hough line/circle transforms and RANSAC robust fitting.
* ``unit3_transforms``     — geometric transformations and image warping.
* ``unit3_stereo``         — stereo vision, depth from disparity, epipolar geometry.
* ``unit4_model_fitting``  — curve fitting, over/under-fitting, regularisation, kernels.
* ``unit5_learning``       — convolution kernels, optimisers, a from-scratch MLP.
* ``unit6_motion``         — optical flow, translational motion estimation, change detection.
* ``unit7_representation`` — PCA representation learning, embeddings, segmentation, a
                              linear generative model.

Every module exposes a ``demo()`` function and is runnable with
``python -m src.classical.<module>``; ``scripts/run_classical.py`` runs them all.
"""

__all__ = [
    "unit1_image_basics",
    "unit2_sampling",
    "unit2_features",
    "unit2_hough_ransac",
    "unit3_transforms",
    "unit3_stereo",
    "unit4_model_fitting",
    "unit5_learning",
    "unit6_motion",
    "unit7_representation",
]
