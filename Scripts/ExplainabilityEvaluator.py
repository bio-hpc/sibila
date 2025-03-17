import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
import pandas as pd
import sys
from pathlib import Path


def load_attributions(folder):
    attr_file_prefix = {
        'Dice': '_Dice.csv', 
        'IntegratedGradients': '_IntegratedGradients.csv', 
        'Lime': '_Lime.csv', 
        'PermutationImportance': '_PermutationImportance.csv', 
        'RFPermutationImportance': '_RFPermutationImportance.csv', 
        'Shapley': '_Shapley.csv'
    }

    folder = Path(folder)
    df_global = {}
    columns = None
    
    for key, suffix in attr_file_prefix.items():
        foo_list = list(folder.glob(f"*{suffix}"))
        
        if len(foo_list) == 1:
            foo = foo_list[0]
            df = pd.read_csv(foo)
            
            if columns is None:
                columns = df['feature'].values
            
            df_global[key] = df['attribution'].values
    
    df_final = pd.DataFrame.from_dict(df_global, orient='index', columns=columns)
    return df_final.to_numpy(), df_final.columns.to_numpy()

# Input parameters
model_prefix = sys.argv[1]
folder = sys.argv[2]

# Global variables
consensus_files = {
    'Average Mean': f'{folder}/Consensus/{model_prefix}_Average_mean.csv',
    'Geometric Mean': f'{folder}/Consensus/{model_prefix}_Geometric_mean.csv',
    'Harmonic Mean': f'{folder}/Consensus/{model_prefix}_Harmonic_mean.csv',
    'Majority Vote': f'{folder}/Consensus/{model_prefix}_Voting.csv',
    'Relative Position': f'{folder}/Consensus/{model_prefix}_Average_ranking.csv',
    'Custom': f'{folder}/Consensus/{model_prefix}_Consensus.csv',
}
results = {}
attributions, feature_names = load_attributions(folder)  # original attributions

for name, filename in consensus_files.items():
    consensus_attrib = pd.read_csv(filename, index_col=0).values.flatten()
    
    if consensus_attrib.shape[0] < len(feature_names):
        n_missing_cols = len(feature_names) - consensus_attrib.shape[0]
        #print(f"Faltan {n_missing_cols} filas")
        consensus_attrib.resize((len(feature_names),))

    # Consistency
    spearman_list = [stats.spearmanr(attributions[i], consensus_attrib)[0] for i in range(attributions.shape[0])]
    avg_spearman = np.mean(spearman_list)
    js_div_list = [jensenshannon(attributions[i], consensus_attrib) for i in range(attributions.shape[0])]
    avg_js_div = np.mean(js_div_list)

    # Stability
    noise = np.random.normal(0, 0.1, size=attributions.shape)
    noisy_attributions = attributions + noise  # slight perturbation
    noisy_consensus_attrib = consensus_attrib
    mad_stability = mean_absolute_error(consensus_attrib, noisy_consensus_attrib)
    variability = np.var(noisy_consensus_attrib - consensus_attrib)
  
    # Fidelity
    fidelity_infidelity = np.mean(np.abs(consensus_attrib - np.mean(attributions, axis=0)))
    dispersion = np.std(consensus_attrib)
    
    results[name] = {
        'Consistency': {'Avg Spearman': avg_spearman, 'Avg JS Divergence': avg_js_div},
        'Stability': {'MAD Stability': mad_stability, 'Variability': variability},
        'Fidelity': {'Infidelity': fidelity_infidelity, 'Dispersion': dispersion}
    }

# Mostrar resultados
for method, metrics in results.items():
    print(f"\nConsensus Function: {method}")
    for category, values in metrics.items():
        print(f"  {category}:")
        for metric, value in values.items():
            print(f"    {metric}: {value:.4f}")
