# MLOps Production-Ready Machine Learning Project

## Overview
This project demonstrates the end-to-end implementation of a machine learning solution designed for production readiness. The pipeline includes data preprocessing, feature engineering, model training, and deployment using modern MLOps tools and best practices.

---

## Prerequisites

- [Anaconda](https://www.anaconda.com/)
- [VS Code](https://code.visualstudio.com/download)
- [Git](https://git-scm.com/)
- [Evidently AI](https://www.evidentlyai.com/) (MLOps tool)
- [MongoDB](https://account.mongodb.com/account/login)

### Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/moro23/easyvisa-dataset).

---

## Git Commands
```bash
git add .
git commit -m "Updated"
git push origin main
```

---

## Environment Setup

### Create and Activate Virtual Environment
```bash
conda create -n visa python=3.8 -y
conda activate visa
```

### Install Required Libraries
```bash
pip install -r requirements.txt
```

---

## Project Workflow

1. **Constants**
2. **Entity Definitions**
3. **Component Development**
4. **Pipeline Implementation**
5. **Main File Execution**

---

## Export Environment Variables
```bash
export MONGODB_URL="mongodb+srv://<username>:<password>@cluster0.mongodb.net/visa"
export STORAGE_CONNECTION_STRING="<Azure Blob Storage connection string>"
```

---

## Azure CICD Deployment with GitHub Actions

This project utilizes GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD) to Azure.

### Save pass: [example]
```bash
vUR02UVhwBnoqxQ3bOhxQaaQ0ZqdKwDOUnllP4LK1s+ACRCeZlol
```

### Deployment Steps
1. **Build the Docker Image**: The GitHub Action builds a Docker image from the source code.

### Test Deployment
1. Commit and push your changes to the `main` branch.
2. GitHub Actions will trigger the workflow, building the Docker image and deploying it to Azure.
3. Monitor the status in the **Actions** tab of your GitHub repository.

### Monitor and Maintain
- **Monitor Deployments**: Use Azure Monitor to track your application's performance and errors.
- **Update Models**: Push new changes to the repository to automatically trigger the CI/CD pipeline for updates.

---


