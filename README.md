## SIBILA
SIBILA Server takes advantage of HPC and ML/DL to provide users with a powerful predictive tool. Several ML models are available and a large set of configuration parameters facilitate the configuration of the tasks. In addition, the server applies the concept of explainable artificial intelligence (XAI) to present the results in a way that users will be able to understand. A collection of interpretability approaches are implemented to identify the most relevant features that were taken into consideration by the model in order to make the prediction. 

### Installation (choose one)
1. git clone https://github.com/bio-hpc/sibila.git
2. git clone git@github.com:bio-hpc/sibila.git
3. gh repo clone bio-hpc/sibila
4. Download the .zip and unzip it in the supercomputing centers you are going to use 

### Download singularity image 
Needed to secure compatibility with all cluster.

cd sibila/Tools/Singularity

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eVI6RpUPvmrOi6Z8p0AeA-dpefdL4UVu&confirm=t' -O sibila.sif

chmod u+x sibila.sif

### Available ML/DL Models and Algorithms
1. **DT (Decision Tree)**
2. **RF (Random Forest)**
3. **SVM (Support Vector Machines)**
4. **XGBOOST (eXtreme Gradient BOOSTing)**
5. **ANN (Artificial Neural Networks)**
6. **KNN (K-Nearest Neighbours)**
7. **RLF (RuLeFit)**
8. **RP (RIPPERk)**

### Available Interpretability Methods
1. **Permutation Feature Importance**
2. **Local Interpretable Model-agnostic Explanations (LIME)**
3. **Integrated Gradients** 
4. **Shapley value**
5. **Diverse Counterfactual Explanations (DICE)**
6. **Partial Dependence Plots (PDP)**
7. **Accumulated Local Effects (ALE)**
8. **Anchors**
9. **Random-Forest based Permutation Feature Importance**

### Scripts
It is a directory that contains scripts for creating random datasets, running manual grid search and joining results into a single output file. 

It is recommended to use these scripts with the SIBILA singularity image "Tools / Singularity / sibila.sif". 
For instance:

singularity exec Tools/Singularity/sibila.sif python3 Scripts/ResultAnalyzer.py -d folder_containing_results -o myfile.xlsx

### CHANGELOG
**v1.2.0** ()
- Added new parameter: --skip-dataset-analysis.
- Use of environment variables in Python code.
- Pass environment variables dynamically to the jobs when parallelizing interpretability.
- Renamed h5 and sif files to use the standard notation.
- Added new parameter: --skip-interpretability.
- Always save execution status in a pickle file.
- Added new parameter: -e, --explanation. Useful when explaining previously trained models.
- Implemented GPU support through Singularity.
- Fix on RandomOversample. Set sampling_strategy=auto.
- Incorporated extra datasets.
- Bind Singularity for executions from outside /home.
- Intrepretability algorithms bulk the attribution of all variables into csv files.
- Reworked explainers.
- Added anchors and RF-based permutation importance as explainers.
- Implemented RIPPERk model's grid search.
- Save the probability of being classified as class X into csv files.
- Extra ranges (10-step) in class probability plot.
- The last column has to be removed from the dataset when using prediction mode (-m).
- Allow text IDs in the first column.

**v1.1.0** (03/09/2022)
- Grid and random search on Artificial Neural Networks.
- Uploaded synthetic datasets for testing.
- Parallelization of interpretability tasks.
- Plot times in logarithmic scale for a better reading.
- Exclusion of Keras Tuner and jobs dir from the compressed file.
- Fixed cross validation with Artificial Neural Networks. Didn't work properly.
- Renamed and standarized metric keys.
- Added mininum number of layers in Artificial Neural Networks.
- Corrected 2-unit layers and dropout layers. They were added after every hidden layer. 
- Added callbacks to speed up Artificial Neural Networks training.
- Collect and plot loss through epochs manually.

**v1.0.0** (30/06/2022)
Initial version
