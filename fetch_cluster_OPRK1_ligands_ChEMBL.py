"""
Author: Yazan Meqbil
Date: 05-01-2023

Description
This script is used to fetch the OPRK1 active compounds from ChEMBL and calculate 
their physicochemical properties. The compounds are then clustered based on their scaffold diversity. 
The results are saved to a CSV file.

"""
!pip install chembl_webresource_client rdkit pandas seaborn matplotlib
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Scaffolds
from chembl_webresource_client.new_client import new_client
import seaborn as sns
import matplotlib.pyplot as plt
import os
from google.colab import drive
from rdkit.ML.Cluster import Butina

# Mount Google Drive (for Google Colab)
# drive.mount('/content/gdrive')

# Set up ChEMBL API client
client = new_client()

def fetch_oprk_smiles():
    # Define target ID and activity threshold
    target_id = 'CHEMBL237'

    # Fetch compounds with activity data for the target
    activity = client.activity
    res = activity.filter(target_chembl_id=target_id, standard_type='IC50', standard_relation='=')

    # Convert results to DataFrame
    df = pd.DataFrame.from_records(res)

    # Convert pchembl_value column to float
    df['pchembl_value'] = df['pchembl_value'].astype(float)

    # Filter compounds with SMILES data
    df = df[df['canonical_smiles'].notnull()]

    # Fetch SMILES strings for compounds
    compounds = client.molecule
    smiles = []
    for cmpd in df['molecule_chembl_id']:
        try:
            compound = compounds.get(cmpd)
            smiles.append(compound['molecule_structures']['canonical_smiles'])
        except KeyError:
            pass

    return smiles

def standardize_and_calc_props(smiles):
    # Define properties to calculate
    properties = [
        'MolWt',
        'HeavyAtomMolWt',
        'ExactMolWt',
        'NumValenceElectrons',
        'NumHDonors',
        'NumHAcceptors',
        'NumRotatableBonds',
        'NumAromaticRings',
        'FractionCSP3'
    ]

    # Initialize DataFrame to store results
    data = pd.DataFrame(columns=properties)

    # Process each compound
    for smi in smiles:
        # Convert SMILES string to RDKit molecule
        mol = Chem.MolFromSmiles(smi)

        # Standardize molecule
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

        # Calculate properties
        row = {}
        for prop in properties:
            row[prop] = Descriptors.MolWt(mol) if prop == 'MolWt' else getattr(Descriptors, prop)(mol)
        data = data.append(row, ignore_index=True)

    return data

    def cluster_scaffolds(data, cutoff=0.4):
    # Generate Murcko scaffolds for each molecule
    scaffolds = []
    for smi in data.index:
        mol = Chem.MolFromSmiles(smi)
        scaffold = Scaffolds.MurckoScaffold.MakeScaffoldGeneric(mol)
        scaffolds.append(scaffold)

    # Convert scaffolds to binary fingerprints
    fps = [Chem.RDKFingerprint(scaffold) for scaffold in scaffolds]

    # Cluster scaffolds based on Tanimoto similarity
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])

    # Perform hierarchical clustering using the Butina method
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

    # Assign a cluster ID to each compound
    cluster_ids = {}
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_ids[data.index[idx]] = i

    # Add cluster ID column to DataFrame
    data['ClusterID'] = data.index.map(cluster_ids)

    return data

# Fetch SMILES strings for OPRK-active compounds from ChEMBL
smiles = fetch_oprk_smiles()

# Calculate physicochemical properties and standardize the dataset
data = standardize_and_calc_props(smiles)

# Generate a histogram of molecular weights
sns.histplot(data['MolWt'], bins=20)
plt.xlabel('Molecular Weight')
plt.ylabel('Count')
plt.title('Distribution of Molecular Weights')
plt.show()

# Cluster the compounds based on their scaffold diversity
clustered_data = cluster_scaffolds(data)

# Save the results to files in a subdirectory of the Google Drive
outdir = '/content/gdrive/MyDrive/OPRK'
if not os.path.exists(outdir):
    os.mkdir(outdir)
data.to_csv(os.path.join(outdir, 'oprk_data_preprocessed.csv'), index=False)
clustered_data.to_csv(os.path.join(outdir, 'oprk_data_processed.csv'), index=False)

# Print confirmation message
print(f'Results saved to {os.path.abspath(outdir)}')