import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diff_cross_attn import Path2SubDifferCrossMHA, Drug2PathDifferCrossMHA
from modules.cross_attn import Path2SubCrossMHA, Drug2PathCrossMHA
from modules.ffn_layer import CelllineFFN, DrugFFN
from modules.mlp_head import DeepRegressorHead
    
# =============================================================================
# DRUG RESPONSE MODEL CLASS
# =============================================================================
class DrugResponseModel(nn.Module):
    """
    Main architecture for Drug Response Prediction.
    """
    def __init__(self, pathway_gene_indices,
                 gene_ffn_output_dim, drug_ffn_output_dim, 
                 cross_attn_dim, final_dim,
                 max_gene_slots, gene_input_dim, drug_input_dim=768, isDiffer=True,
                 gene_ffn_hidden_dim=1024, drug_ffn_hidden_dim=1024,
                 gene_ffn_dropout=0.5, drug_ffn_dropout=0.5,
                 num_heads=4, depth=2, mlp_dropout=0.3, final_dim_reduction_factor=2,
                 # DeepRegressorHead parameters
                 regressor_hidden_dims=[512, 512, 512, 256, 256],
                 regressor_dropout=0.15,
                 regressor_norm="batch",
                 regressor_act="gelu",
                 regressor_residual_every=2,
                 regressor_residual_proj=True,
                 regressor_last_dropout=False,
                 regressor_final_norm=False,
                 # Flexible configuration for Regressor (List types)
                 regressor_dropouts=None,
                 regressor_norms=None,
                 regressor_acts=None):
        super(DrugResponseModel, self).__init__()
        
        # 1. Register Pathway Mapping (Indices) as a buffer (not a learnable parameter)
        self.register_buffer('pathway_gene_indices', pathway_gene_indices)
        self.max_gene_slots = max_gene_slots
        self.gene_value_norm = nn.LayerNorm([94, gene_ffn_output_dim])
        
        # 2. Initialize Feed-Forward Networks (FFNs) for Feature Encoding
        # Encodes Cell-line (Gene) features
        self.gene_ffn = CelllineFFN(input_dim=gene_input_dim, output_dim=gene_ffn_output_dim, 
                                       hidden_dim=gene_ffn_hidden_dim, dropout_rate=gene_ffn_dropout)
        # Encodes Global Drug features
        self.drug_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                                         hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
        # Encodes Drug Substructure features
        self.sub_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                                         hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
                
        # 3. Initialize Cross-Attention Modules
        # isDiffer=True: Uses Differential Cross-Attention (Proposed Method)
        # isDiffer=False: Uses Standard Cross-Attention (Baseline)
        if isDiffer:
            # View 1
            self.Path2Drug_cross_attention = Path2SubDifferCrossMHA(
                pathway_embed_dim=gene_ffn_output_dim, 
                drug_embed_dim=drug_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
            # View 2
            self.Drug2Path_cross_attention = Drug2PathDifferCrossMHA(
                drug_embed_dim=drug_ffn_output_dim, 
                pathway_embed_dim=gene_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
        else:
            self.Path2Drug_cross_attention = Path2SubCrossMHA(
                pathway_embed_dim=gene_ffn_output_dim, 
                drug_embed_dim=drug_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
            self.Drug2Path_cross_attention = Drug2PathCrossMHA(
                drug_embed_dim=drug_ffn_output_dim, 
                pathway_embed_dim=gene_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
            
        # 4. Final Regression Head (MLP)
        # Input: Concatenation of (Pathway + Drug + Substructure) features
        self.head = DeepRegressorHead(
            input_dim=3 * cross_attn_dim,
            hidden_dims=regressor_hidden_dims,
            dropout=regressor_dropouts if regressor_dropouts is not None else regressor_dropout,
            norm=regressor_norms if regressor_norms is not None else regressor_norm,
            act=regressor_acts if regressor_acts is not None else regressor_act,
            residual_every=regressor_residual_every,
            residual_proj=regressor_residual_proj,
            out_dim=1,
            last_dropout=regressor_last_dropout,
            final_norm=regressor_final_norm
        )

    def forward(self, gene_embeddings_input, drug_embeddings_input, drug_substructure_embeddings, drug_multitoken_masks):
        """
        Forward pass of the model.
        
        Args:
            gene_embeddings_input: [Batch, Total_Genes]
            drug_embeddings_input: [Batch, Drug_Dim] (Global feature)
            drug_substructure_embeddings: [Batch, Max_Sub, Drug_Dim] (Local features)
            drug_multitoken_masks: [Batch, Max_Sub]
        """
        B = gene_embeddings_input.size(0)
        P = self.pathway_gene_indices.size(0)     # Number of Pathways

        # ---------------------------------------------------------------------
        # [Step 1] Pathway Construction (Gene Mapping)
        # ---------------------------------------------------------------------
        # gene_embeddings_input: [B, G]
        genes = gene_embeddings_input              # [B, G]
        indices = self.pathway_gene_indices.to(torch.long)      # [P, G_p]

        # Create mask for padding indices (-1)
        pad_mask = (indices == -1)
        safe_idx = indices.clone()
        safe_idx[pad_mask] = 0

        # Expand dimensions for gathering
        expanded_genes   = genes.unsqueeze(1).expand(-1, P, -1)         # [B, P, G]
        expanded_indices = safe_idx.unsqueeze(0).expand(B, -1, -1)      # [B, P, G_p]

        # Gather actual gene values based on pathway indices
        pathway_specific_genes = torch.gather(
            expanded_genes, 2, expanded_indices.to(expanded_genes.device)
        )  # [B, P, G_p]

        # Apply mask: Set padded gene values to 0.0
        pathway_specific_genes = pathway_specific_genes.masked_fill(
            pad_mask.unsqueeze(0).to(pathway_specific_genes.device), 0.0
        )  # [B, P, G_p]

        # ---------------------------------------------------------------------
        # [Step 2] Feature Encoding (FFN)
        # ---------------------------------------------------------------------
        # 1. Gene FFN Module
        ffn_output = self.gene_ffn(pathway_specific_genes)
        gene_embedded_value = ffn_output.view(B, P, -1)
        pathway_embeddings = gene_embedded_value
        pathway_embeddings = self.gene_value_norm(gene_embedded_value)  # [B, P, gene_ffn_output_dim]

        # 2. Multi-Token Drug FFN Module
        sub_embeddings = self.sub_ffn(drug_substructure_embeddings, drug_multitoken_masks)  # [B, L, drug_ffn_output_dim]
        drug_embeddings_input = drug_embeddings_input.unsqueeze(1)  # [B, drug_ffn_output_dim] -> [B, 1, drug_ffn_output_dim]
        drug_embeddings = self.drug_ffn(drug_embeddings_input)  # [B, 768] -> [B, 1, drug_ffn_output_dim]

        # ---------------------------------------------------------------------
        # [Step 3] Dual-View Cross Attention
        # ---------------------------------------------------------------------
        # View 1: Path2Sub
        path2drug_out, path2drug_weights = self.Path2Drug_cross_attention(
            query=pathway_embeddings,    # [B, P, gene_ffn_output_dim]
            key=sub_embeddings,         # [B, L, drug_ffn_output_dim]
            key_mask=drug_multitoken_masks  # [B, L]
        )

        # View 2: Drug2Path 
        drug2path_out, drug2path_weights = self.Drug2Path_cross_attention(
            query=drug_embeddings,       # [B, 1, drug_ffn_output_dim]
            key=pathway_embeddings,      # [B, P, gene_ffn_output_dim]
        )
        
        # ---------------------------------------------------------------------
        # [Step 4] Feature Aggregation & Prediction
        # ---------------------------------------------------------------------
        # Mean pooling for pathway embeddings
        final_pathway_embedding = path2drug_out.mean(dim=1)
        final_drug_embedding = drug2path_out.mean(dim=1)
        final_sub_embedding = sub_embeddings.mean(dim=1)

        # Concatenate embeddings and MLP
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding, final_sub_embedding), dim=-1)
        
        # Final Regression
        y = self.head(combined_embedding)  # [B, 1]

        return y, path2drug_weights, drug2path_weights, pathway_embeddings