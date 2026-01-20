import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

# =============================================================================
#  [1] DATASET CLASS
# =============================================================================
class DrugResponseDataset(Dataset):
    """Dataset for Drug Response Prediction"""

    def __init__(self, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels, sample_indices, **kwargs):  
        self.gene_embeddings = gene_embeddings
        self.drug_embeddings = drug_embeddings
        self.drug_substructure_embeddings = drug_substructure_embeddings 
        self.drug_substructure_masks = drug_substructure_masks
        self.labels = labels
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        # 1. Get indices
        cell_line_id, drug_id = self.sample_indices[idx]

        # 2. Get features
        gene_embedding = self.gene_embeddings[cell_line_id]  
        drug_embedding = self.drug_embeddings[drug_id]  
        drug_substructure_embedding = self.drug_substructure_embeddings[drug_id]  
        drug_substructure_mask = self.drug_substructure_masks[drug_id]  
        label = self.labels[cell_line_id, drug_id]

        return {
            'gene_embedding': gene_embedding,
            'drug_embedding': drug_embedding,
            'drug_substructure_embedding': drug_substructure_embedding,
            'drug_substructure_mask': drug_substructure_mask,
            'label': label,
            'sample_index': (cell_line_id, drug_id) 
        }
    
# =============================================================================
#  [2] COLLATE FUNCTION
# =============================================================================
def collate_fn(batch):
    """Stacks samples into a batch."""
    gene_embeddings = []
    drug_embeddings = []
    drug_substructure_embeddings = []
    drug_substructure_masks = []
    labels = []
    sample_indices = []
    
    for item in batch:
        gene_embeddings.append(item['gene_embedding'])
        drug_embeddings.append(item['drug_embedding'])
        drug_substructure_embeddings.append(item['drug_substructure_embedding'])
        drug_substructure_masks.append(item['drug_substructure_mask'])
        labels.append(item['label'])
        sample_indices.append(item['sample_index'])

    return {
        'gene_embeddings': torch.stack(gene_embeddings), 
        'drug_embeddings': torch.stack(drug_embeddings),  
        'drug_substructure_embeddings': torch.stack(drug_substructure_embeddings),  
        'drug_substructure_masks': torch.stack(drug_substructure_masks),  
        'labels': torch.tensor(labels, dtype=torch.float32),              
        'sample_indices': sample_indices
    }