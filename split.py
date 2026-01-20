import torch
from itertools import product
from sklearn.model_selection import train_test_split

# =============================================================================
# [1] HELPER FUNCTION
# =============================================================================
def filter_data(sample_indices, gene_embeddings, drug_embeddings, drug_graphs, drug_masks, labels_dict):
    """
    Utility function to extract data subsets corresponding to specific indices.
    """
    filtered_gene_embeddings = {cell_line: gene_embeddings[cell_line] for cell_line, _ in sample_indices}
    filtered_drug_embeddings = {drug: drug_embeddings[drug] for _, drug in sample_indices}
    filtered_drug_graphs = {drug: drug_graphs[drug] for _, drug in sample_indices}
    filtered_drug_masks = {drug: drug_masks[drug] for _, drug in sample_indices}
    filtered_labels = {(cell_line, drug): labels_dict[(cell_line, drug)] for cell_line, drug in sample_indices}

    return filtered_gene_embeddings, filtered_drug_embeddings, filtered_drug_graphs, filtered_drug_masks, filtered_labels

# =============================================================================
# [2] DATA LOADING
# =============================================================================
# Load pre-processed embeddings and label dictionaries
gene_embeddings = torch.load('./input/gene_embeddings_10_fold_binary.pt') 
drug_embeddings = torch.load('./input/drug_embeddings.pt')
drug_substructure_embeddings = torch.load('./input/drug_BRICS_embeddings.pt')
drug_substructure_masks = torch.load('./input/drug_BRICS_masks.pt')
labels_dict = torch.load('./input/response_label_dict_LN.pt')

cell_lines = list(gene_embeddings.keys())
drugs = list(drug_embeddings.keys())
sample_indices = list(labels_dict.keys())

# =============================================================================
# [3] DATA SPLITTING 
# =============================================================================
# Define split ratios (Train:Val:Test = 3:1:1)
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Phase 1: Separate Training set (60%) from the rest
train_indices, temp_indices = train_test_split(
    sample_indices, test_size=(1 - train_ratio), random_state=42
)
# Phase 2: Split the remaining data into Validation (20%) and Test (20%) sets
val_indices, test_indices = train_test_split(
    temp_indices, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
)

# Extract and save the dedicated Test set for final evaluation
test_gene_emb, test_drug_emb, test_drug_graphs, test_masks, test_labels = filter_data(
    test_indices, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels_dict
)

test_dataset = {
    'gene_embeddings': test_gene_emb,
    'drug_embeddings': test_drug_emb,
    'drug_substructure_embeddings': test_drug_graphs,
    'drug_substructure_masks': test_masks,
    'labels': test_labels,
    'sample_indices': test_indices,
}
torch.save(test_dataset, './dataset/test_dataset.pt')

# =============================================================================
# [4] CROSS-VALIDATION (5-Fold)
# =============================================================================
from sklearn.model_selection import KFold

# Initialize 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Combine Train and Validation sets for CV splitting
train_val_indices = train_indices + val_indices

for fold_num, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
    current_train_indices = [train_val_indices[i] for i in train_idx]
    current_val_indices = [train_val_indices[i] for i in val_idx]

    # Extract data for the current fold
    fold_train_gene_emb, fold_train_drug_emb, fold_train_drug_graphs, fold_train_masks, fold_train_labels = filter_data(
        current_train_indices, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels_dict
    )

    fold_val_gene_emb, fold_val_drug_emb, fold_val_drug_graphs, fold_val_masks, fold_val_labels = filter_data(
        current_val_indices, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels_dict
    )

    # Construct the dataset dictionary for the current fold
    fold_dataset = {
        'train': {
            'gene_embeddings': fold_train_gene_emb,
            'drug_embeddings': fold_train_drug_emb,
            'drug_substructure_embeddings': fold_train_drug_graphs,
            'drug_substructure_masks': fold_train_masks,
            'labels': fold_train_labels,
            'sample_indices': current_train_indices,
        },
        'validation': {
            'gene_embeddings': fold_val_gene_emb,
            'drug_embeddings': fold_val_drug_emb,
            'drug_substructure_embeddings': fold_val_drug_graphs,
            'drug_substructure_masks': fold_val_masks,
            'labels': fold_val_labels,
            'sample_indices': current_val_indices,
        }
    }

    torch.save(fold_dataset, f'./dataset/cross_valid_fold_{fold_num + 1}.pt')

print("Test dataset and cross-validation folds have been successfully saved.")
