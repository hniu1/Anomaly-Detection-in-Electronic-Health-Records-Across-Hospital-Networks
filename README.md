# Anomaly-Detection-in-Electronic-Health-Records-Across-Hospital-Networks

This repo contains the code and package we used for the study. In this article, we introduce a new approach to detecting anomalies in EHR data within a network of hospitals. This is achieved by combining advanced machine learning techniques with graph algorithms to create a tool capable of swiftly identifying and responding to deviations.

Note: we are updating the repo and preparing sample data for testing.

![alt](img/Figure1.pdf)

## Installation

To set up the required environment, follow these steps:

1. Clone the repository.
2. Create a Conda environment using the provided YAML file:

   ```shell script
   conda env create -f ./environment/environment.yml
    ```

Reference: https://docs.conda.io/en/latest/miniconda.html

## Data preprocessing and voting machine

We provide the data preprocessing and voting machine steps in 0VM_AllStation_P.py file. 

## Graph Construction

File 1BuildGraph.py shows the steps of Graph construction we mentioned in the paper.

## System-level network anomaly detection

In 2PatchEvaluation, graph decomposition steps are included with the two packages we used:

* Thresholding [1].
* Paraclique [2]


## References:
[1] Bleker, Carissa, Stephen K. Grady, and Michael A. Langston. "A Comparative Study of Gene Co-Expression Thresholding Algorithms." Journal of Computational Biology (2024).

[2] Hagan, Ronald D., Michael A. Langston, and Kai Wang. "Lower bounds on paraclique density." Discrete Applied Mathematics 204 (2016): 208-212.
