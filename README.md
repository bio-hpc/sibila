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

wget --no-check-certificate -r "https://bio-hpc.ucam.edu/owncloud/index.php/s/zgxfq8ao5Z4nlUV" -O sibila.simg

chmod u+x sibila.simg

#unzip singularity/singularity.zip -d singularity/

#rm singularity/singularity.zip

### Available ML/DL Models and Algorithms
1. **DT (Decision Tree)**
2. **RF (Random Forest)**
3. **SVM (Support Vector Machines)**
4. **XGBOOST (eXtreme Gradient BOOSTing)**
5. **ANN (Artificial Neural Networks)**
6. **KNN (K-Nearest Neighbours)**
7. **RLF (RuLEFit)**
8. **RP (RiPper)**

### Available Interpretability Methods
1. **Permutation Feature Importance**
2. **Local Interpretable Model-agnostic Explanations (LIME)**
3. **Integrated Gradients** 
4. **Shapley value**
5. **Diverse Counterfactual Explanations (DICE)**
6. **Partial Dependence Plots (PDP)**
7. **Accumulated Local Effects (ALE)**

### Scripts
It is a directory that contains scripts for creating random datasets, running manual grid search and joining results into a single output file. 

It is recommended to use these scripts with the SIBILA singularity image "Tools / Singularity / sibila.simg". 
For instance:

singularity exec Tools/Singularity/sibila.simg python3 Scripts/ResultAnalyzer.py -d folder_containing_results -o myfile.xlsx
