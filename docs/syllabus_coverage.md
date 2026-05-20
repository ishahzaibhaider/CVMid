# Syllabus coverage — AIC341 Introduction to Computer Vision

This project deliberately spans the **entire AIC341 course outline**. The
VeriVision deep-learning detector is the flagship application (Units 5 & 7);
the `src/classical/` toolkit adds a runnable demonstration for every other
unit. Every module writes real result figures to `reports/figures/classical/`.

## Unit-by-unit map

| Unit | Syllabus topics | Module(s) | Figures produced |
|---|---|---|---|
| **1** | Image formation, geometric vision, the CV pipeline, applications | `unit1_image_basics` | image formation, channels & histogram, colour spaces, CV pipeline |
| **2** | Sampling & aliasing, up/down-sampling, multiscale pyramids, edges, corners, blobs, local features, Hough transform, RANSAC | `unit2_sampling`, `unit2_features`, `unit2_hough_ransac` | aliasing, resampling, pyramids, edges, corners, blobs, ORB keypoints, Hough lines, Hough circles, RANSAC |
| **3** | Image transformations & warping, stereo vision, depth from disparity, epipolar geometry, rectification | `unit3_transforms`, `unit3_stereo` | transformation hierarchy, warping, stereo pair, disparity & depth, epipolar geometry |
| **4** | Model fitting, regularisation, kernel regression, polynomial curve fitting, over/under-fitting | `unit4_model_fitting` | polynomial fits, bias-variance curve, ridge regularisation, kernel regression |
| **5** | Supervised learning, neural networks, SGD, optimisers, back-propagation, CNNs & architectures | `unit5_learning` + `src/models/` (custom CNN, ResNet-50) | convolution kernels, optimiser comparison, from-scratch MLP + the trained CNN/ResNet results |
| **6** | Motion estimation, translational alignment, optical flow, change detection | `unit6_motion` | optical flow, phase-correlation motion estimation, change detection |
| **7** | Representation learning, perceptual grouping, generative models, conditional generative models | `unit7_representation` + VeriVision itself | PCA eigen-images, 2-D embeddings, segmentation, a linear generative model; VeriVision *detects* conditional generative models |

## Lab CLO mapping

| CLO | Statement | Where it is evidenced |
|---|---|---|
| **CLO-5** | Implement core CV algorithms using appropriate tools and libraries | The 10 modules in `src/classical/` + the VeriVision training/eval pipeline |
| **CLO-6** | Develop a CV application in a team environment | VeriVision end-to-end: dataset → preprocessing → models → evaluation → Streamlit demo, built by a 3-member team |

## Reproducing every figure

```bash
python -m scripts.run_classical --unit all     # all classical-CV figures
python -m scripts.verivision_figures           # VeriVision data figures
# DL result figures come from scripts/evaluate.py after training on Kaggle
```

See [`../README.md`](../README.md) for full setup and run instructions.
