# Point2Building: Reconstructing Buildings from Airborne LiDAR Point Clouds

## Introduction

Welcome to the code repository for the paper **"Point2Building: Reconstructing Buildings from Airborne LiDAR Point Clouds"**. This repository contains the implementation of our learning-based approach to reconstruct 3D polygonal meshes of buildings from airborne LiDAR point clouds.

### Abstract

We present a learning-based approach to reconstruct buildings as 3D polygonal meshes from airborne LiDAR point clouds. What makes 3D building reconstruction from airborne LiDAR hard is the large diversity of building designs and especially roof shapes, the low and varying point density across the scene, and the often incomplete coverage of building facades due to occlusions by vegetation or to the viewing angle of the sensor. To cope with the diversity of shapes and inhomogeneous and incomplete object coverage, we introduce a generative model that directly predicts 3D polygonal meshes from input point clouds. Our autoregressive model, called Point2Building, iteratively builds up the mesh by generating sequences of vertices and faces. This approach enables our model to adapt flexibly to diverse geometries and building structures. Unlike many existing methods that rely heavily on pre-processing steps like exhaustive plane detection, our model learns directly from the point cloud data, thereby reducing error propagation and increasing the fidelity of the reconstruction. We experimentally validate our method on a collection of airborne LiDAR data of Zurich, Berlin, and Tallinn. Our method shows good generalization to diverse urban styles.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Point2Building.git
   cd Point2Building
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate 
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the vertex generation model, run:
    ```bash
    python train_vertex_model.py
    ```

To train the face generation model, run:
    ```bash
    python train_face_model.py
    ```

### Testing

To test the trained models and generate 3D polygonal meshes, run:
    ```bash
    python test_models.py
    ```

### Visualization

To visual the results of the region, run:
    ```bash
    python visualize_city.py
    ```

## Downloads

For training and testing, you can download the following resources from [link](https://drive.google.com/file/d/1DJRQ6uwvFmM4fiaoeURsxSYjTvPcJIym/view?usp=share_link)
- Processed Dataset: A ready-to-use dataset specifically processed for this project.
- Pre-trained Models: Pre-trained models to facilitate immediate testing and evaluation.
- Visualized Results: Examples of visualized results from the models for reference.

## Acknowledgements

This work is built upon [polygen](https://github.com/anshulcgm/polygen/tree/master). We thank the author's great work and repo.
