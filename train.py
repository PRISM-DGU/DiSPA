import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
import logging
import time
from tqdm import tqdm
from datetime import datetime
import os
import yaml
import argparse
from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, set_seed
import shutil

# =============================================================================
# [1] SCHEDULER FACTORY
# =============================================================================
def create_scheduler(optimizer, scheduler_config, num_epochs, steps_per_epoch):
    """Create LR scheduler based on config"""
    scheduler_type = scheduler_config.get('type', None)

    if scheduler_type is None or scheduler_type == 'null':
        return None, 'none'

    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'onecycle':
        params = scheduler_config['onecycle']
        scheduler = OneCycleLR(
            optimizer,
            max_lr=params['max_lr'],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=params['pct_start'],
            anneal_strategy=params['anneal_strategy'],
            div_factor=params['div_factor'],
            final_div_factor=params['final_div_factor']
        )
        step_type = 'batch'  # step per batch

    elif scheduler_type == 'plateau':
        params = scheduler_config['plateau']
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=params['mode'],
            factor=params['factor'],
            patience=params['patience'],
            min_lr=params['min_lr'],
            threshold=params['threshold']
        )
        step_type = 'epoch_metric'  # step per epoch with metric

    elif scheduler_type == 'cosine_warm':
        params = scheduler_config['cosine_warm']
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params['T_0'],
            T_mult=params['T_mult'],
            eta_min=params['eta_min']
        )
        step_type = 'epoch'  # step per epoch

    elif scheduler_type == 'cyclic':
        params = scheduler_config['cyclic']
        scheduler = CyclicLR(
            optimizer,
            base_lr=params['base_lr'],
            max_lr=params['max_lr'],
            step_size_up=params['step_size_up'],
            mode=params['mode'],
            gamma=params.get('gamma', 1.0),
            cycle_momentum=False  # For Adam
        )
        step_type = 'batch'  # step per batch

    else:
        logging.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
        return None, 'none'

    logging.info(f"Created {scheduler_type} scheduler with step_type={step_type}")
    return scheduler, step_type

# =============================================================================
# [2] CONFIGURATION & SETUP
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Drug Response Prediction Training')
    parser.add_argument('--config', type=str, default='config.yml', 
                       help='Path to configuration file (default: config.yml)')
    return parser.parse_args()

def load_config(config_path='config.yml'):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def parse_regressor_config(model_config):
    """Parse regressor configuration"""
    # Check for new layer_configs style
    if 'regressor_layer_configs' in model_config:
        layer_configs = model_config['regressor_layer_configs']
        
        hidden_dims = [layer['dim'] for layer in layer_configs]
        dropouts = [layer.get('dropout', 0.1) for layer in layer_configs]
        norms = [layer.get('norm', 'batch') for layer in layer_configs]
        acts = [layer.get('act', 'gelu') for layer in layer_configs]
        
        return {
            'regressor_hidden_dims': hidden_dims,
            'regressor_dropouts': dropouts,  
            'regressor_norms': norms,        
            'regressor_acts': acts,          
            'regressor_residual_every': model_config.get('regressor_residual_every', 2),
            'regressor_residual_proj': model_config.get('regressor_residual_proj', True),
            'regressor_last_dropout': model_config.get('regressor_last_dropout', False),
            'regressor_final_norm': model_config.get('regressor_final_norm', False)
        }
    
    # Legacy style support
    else:
        DEFAULT_REGRESSOR_DIMS = [512, 512, 512, 256, 256]
        explicit_hidden_dims = model_config.get('regressor_hidden_dims')
        single_hidden_dim = model_config.get('regressor_hidden_dim')
        layer_count = model_config.get('regressor_layer_count')
        
        if single_hidden_dim is not None or layer_count is not None:
            base_dim = single_hidden_dim if single_hidden_dim is not None else (
                explicit_hidden_dims[0] if explicit_hidden_dims else DEFAULT_REGRESSOR_DIMS[0]
            )
            resolved_layer_count = layer_count if layer_count is not None else (
                len(explicit_hidden_dims) if explicit_hidden_dims else len(DEFAULT_REGRESSOR_DIMS)
            )
            regressor_hidden_dims = [base_dim for _ in range(resolved_layer_count)]
        else:
            regressor_hidden_dims = explicit_hidden_dims if explicit_hidden_dims else DEFAULT_REGRESSOR_DIMS
        
        return {
            'regressor_hidden_dims': regressor_hidden_dims,
            'regressor_dropout': model_config.get('regressor_dropout', 0.15),
            'regressor_norm': model_config.get('regressor_norm', 'batch'),
            'regressor_act': model_config.get('regressor_act', 'gelu'),
            'regressor_residual_every': model_config.get('regressor_residual_every', 2),
            'regressor_residual_proj': model_config.get('regressor_residual_proj', True),
            'regressor_last_dropout': model_config.get('regressor_last_dropout', False),
            'regressor_final_norm': model_config.get('regressor_final_norm', False)
        }

# Initial Setup
args = parse_args()
config = load_config(args.config)

# System Settings
os.environ['TZ'] = config['system']['timezone']
time.tzset()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config['system']['pytorch_cuda_alloc_conf']

# Fixed Directory Paths
CHECKPOINT_DIR = "./checkpoints"
PLOT_DIR = "./plots"
LOG_DIR = "log"
WEIGHTS_DIR = "weights"
RESULTS_DIR = "results"

# Device Configuration
device = config['training']['device']
if device == "cuda:0" and not torch.cuda.is_available():
    device = "cpu"
config['training']['device'] = torch.device(device)

# Extract parameters
training_config = config['training']
save_config = config['save']
model_config = config['model']
data_config = config['data']
system_config = config['system']

# Model Hyperparameters
GENE_FFN_OUTPUT_DIM = model_config['gene_ffn_output_dim']
DRUG_FFN_OUTPUT_DIM = model_config['drug_ffn_output_dim'] 
MAX_GENE_SLOTS = model_config['max_gene_slots']
GENE_INPUT_DIM = model_config['gene_input_dim'] 
DRUG_INPUT_DIM = model_config.get('drug_input_dim', 768) 
IS_DIFFER = model_config.get('isDiffer', True) # Toggle for Differential Cross Attention
CROSS_ATTN_EMBEDDING_DIM = model_config['cross_attn_embedding_dim']
FINAL_EMBEDDING_DIM = model_config['final_embedding_dim']
OUTPUT_DIM = model_config['output_dim']

# Parse Regressor Cofig
regressor_config = parse_regressor_config(model_config)

BATCH_SIZE = training_config['batch_size']
FILE_NAME = os.environ.get("RUN_NAME", datetime.now().strftime('%Y%m%d_%H'))
DEVICE = training_config['device']
SAVE_FOLD_NUMBER = save_config['save_fold_number'] 

# Directory Setup
log_dir = f"{LOG_DIR}/{FILE_NAME}"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/train.log"
chpt_dir = f"{CHECKPOINT_DIR}/{FILE_NAME}"
os.makedirs(chpt_dir, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logging.info(
    "Embedding Dimensions and Hidden Layers: "
    f"  GENE_FFN_OUTPUT_DIM: {GENE_FFN_OUTPUT_DIM} "
    f"  DRUG_FFN_OUTPUT_DIM: {DRUG_FFN_OUTPUT_DIM} "
    f"  DRUG_INPUT_DIM: {DRUG_INPUT_DIM} "
    f"  FINAL_DIM: {FINAL_EMBEDDING_DIM} "
    f"  OUTPUT_DIM: {OUTPUT_DIM}")
logging.info(
    "Training Configuration: "
    f"  BATCH_SIZE: {BATCH_SIZE} "
    f"  Learning Rate: {training_config['learning_rate']} "
    f"  Number of Epochs: {training_config['num_epochs']} "
)
logging.info(f"CUDA is available: {torch.cuda.is_available()}")

# Seed Setting
set_seed(system_config['seed'])

# Common Data Load
pathway_gene_indices = torch.load(data_config['pathway_gene_indices_path'])

# =============================================================================
# [3] TRAINING HELPER FUNCTION
# =============================================================================
def process_batch(batch_idx, batch, epoch, model, criterion, device, mode="train", is_best_epoch=False, save_weights=False, save_dir_root=None):
    gene_embeddings = batch['gene_embeddings'].to(device)
    drug_embeddings = batch['drug_embeddings'].to(device) 
    drug_substructure_embeddings = batch['drug_substructure_embeddings'].to(device)  
    drug_substructure_masks = batch['drug_substructure_masks'].to(device)  
    labels = batch['labels'].to(device)
    sample_indices = batch['sample_indices']

    # Forward Pass
    outputs, path2drug_weights, drug2path_weights, pathway_embeddings = model(gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks)
    outputs = outputs.squeeze(dim=-1) 

    # Calculate Loss (MSE)
    loss = criterion(outputs, labels) 
    rmse = torch.sqrt(loss).item()  

    # Save Attention Weights
    if mode == "train" and save_weights and save_config['isSave']:
        save_dir = f"{save_dir_root}/current_epoch"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(path2drug_weights.detach().cpu(), f"{save_dir}/B{batch_idx}_path2drug_weight.pt")
        torch.save(drug2path_weights.detach().cpu(), f"{save_dir}/B{batch_idx}_drug2path_weight.pt")
        torch.save(pathway_embeddings.detach().cpu(), f"{save_dir}/B{batch_idx}_pathway_embeddings.pt")
        torch.save(sample_indices, f"{save_dir}/B{batch_idx}_samples.pt")

    return outputs, loss, rmse, sample_indices, labels

# =============================================================================
# [4] MAIN TRAINING LOOP (5-Fold CV)
# ============================================================================
for fold in range(1, 6):
    logging.info(f"▶ Starting Fold {fold}")

    # Fold Directory Setting
    fold_checkpoint_dir = f"{chpt_dir}/fold_{fold}"
    fold_attn_dir = f"{WEIGHTS_DIR}/{FILE_NAME}/fold_{fold}"
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    os.makedirs(fold_attn_dir, exist_ok=True)
    
    # Fold Data Load & Init Model
    fold_data = torch.load(f'{data_config["cross_validation_data_dir"]}/cross_valid_fold_{fold}.pt')
    
    train_dataset = DrugResponseDataset(**fold_data['train'])
    val_dataset = DrugResponseDataset(**fold_data['validation'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    # Initialize Model    
    model = DrugResponseModel(
        pathway_gene_indices=pathway_gene_indices,
        gene_ffn_output_dim=GENE_FFN_OUTPUT_DIM,
        drug_ffn_output_dim=DRUG_FFN_OUTPUT_DIM,
        cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
        final_dim=FINAL_EMBEDDING_DIM,
        max_gene_slots=MAX_GENE_SLOTS,
        gene_input_dim=GENE_INPUT_DIM,
        drug_input_dim=DRUG_INPUT_DIM,
        isDiffer=IS_DIFFER,
        gene_ffn_hidden_dim=model_config['gene_ffn_hidden_dim'],
        drug_ffn_hidden_dim=model_config['drug_ffn_hidden_dim'],
        gene_ffn_dropout=model_config['gene_ffn_dropout'],
        drug_ffn_dropout=model_config['drug_ffn_dropout'],
        num_heads=model_config['num_heads'],
        depth=model_config['depth'],
        mlp_dropout=model_config['mlp_dropout'],
        final_dim_reduction_factor=model_config['final_dim_reduction_factor'],
        # DeepRegressorHead parameters
        **regressor_config
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0) # <-- weight_decay 추가
    )

    # Create LR Scheduler
    scheduler_config = training_config.get('scheduler', {})
    scheduler, scheduler_step_type = create_scheduler(
        optimizer,
        scheduler_config,
        training_config['num_epochs'],
        len(train_loader)
    )

    # Variables for Monitoring
    train_rmses, val_rmses = [], []
    best_val_rmse = float('inf')
    patience_counter = 0

    # ==========================
    # Epoch Loop
    # ==========================
    for epoch in range(training_config['num_epochs']):
        epoch_start = time.time()
        logging.info(f"Epoch [{epoch+1}/{training_config['num_epochs']}] started.")

        train_actuals, train_predictions, train_samples = [], [], []
        val_actuals, val_predictions, val_samples = [], [], []

        # ----------------------
        # Training Phase
        # ----------------------
        model.train()
        total_train_se, total_train_samples = 0, 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            optimizer.zero_grad()

            # Save weights condition
            save_weights_this_epoch = (fold == SAVE_FOLD_NUMBER) and (epoch + 1 >= 40) and save_config['isSave'] # After 40 Epochs, Specific Fold for File Memorization Save
            
            outputs, loss, rmse, sample_indices, labels  = process_batch(batch_idx, batch, epoch+1, model, criterion, DEVICE, mode="train", is_best_epoch=False, save_weights=save_weights_this_epoch, save_dir_root=fold_attn_dir)
            
            loss.backward()
            optimizer.step()

            # Batch-level scheduler step (OneCycleLR, CyclicLR)
            if scheduler is not None and scheduler_step_type == 'batch':
                scheduler.step()

            # Accumulate metrics
            se = ((outputs.detach() - labels.detach()) ** 2).sum()
            total_train_se += se.item()
            total_train_samples += labels.numel()

            train_actuals.extend(labels.detach().cpu()) # Labels
            train_predictions.extend(outputs.detach().cpu()) # Predictions
            train_samples.extend(sample_indices) # Samples

        train_rmse = (total_train_se / total_train_samples) ** 0.5         
        train_rmses.append(train_rmse)

        # ----------------------
        # Validation Phase
        # ----------------------
        model.eval()
        total_val_se, total_val_samples = 0, 0

        with torch.no_grad():
            for val_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                outputs, loss, rmse, sample_indices, labels = process_batch(val_idx, batch, epoch+1, model, criterion, DEVICE, mode="val", is_best_epoch=False, save_weights=False)
                se = ((outputs.detach() - labels.detach()) ** 2).sum()
                total_val_se += se.item()
                total_val_samples += labels.numel()

                val_actuals.extend(labels.detach().cpu())
                val_predictions.extend(outputs.detach().cpu())
                val_samples.extend(sample_indices)

        val_rmse = (total_val_se / total_val_samples) ** 0.5         
        val_rmses.append(val_rmse)

        # Compute Correlations (PCC & SCC)
        train_actuals_np = torch.stack(train_actuals).cpu().numpy()
        train_predictions_np = torch.stack(train_predictions).detach().cpu().numpy()
        val_actuals_np = torch.stack(val_actuals).cpu().numpy()
        val_predictions_np = torch.stack(val_predictions).detach().cpu().numpy()

        train_pcc = pearsonr(train_actuals_np, train_predictions_np)[0]
        train_scc = spearmanr(train_actuals_np, train_predictions_np)[0]
        val_pcc = pearsonr(val_actuals_np, val_predictions_np)[0]
        val_scc = spearmanr(val_actuals_np, val_predictions_np)[0]

        # Log Metrics
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Fold {fold} Epoch [{epoch+1}/{training_config['num_epochs']}] completed. \n"
                     f"Train RMSE: {train_rmse:.4f}, Train PCC: {train_pcc:.4f}, Train SCC: {train_scc:.4f}\n"
                     f"Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}, Val SCC: {val_scc:.4f}\n"
                     f"Current LR: {current_lr:.2e}"
        )

        # Epoch-level scheduler step
        if scheduler is not None:
            if scheduler_step_type == 'epoch':
                scheduler.step()
            elif scheduler_step_type == 'epoch_metric':
                scheduler.step(val_rmse)

        # ----------------------
        # Checkpointing & Early Stopping
        # ----------------------
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0

            # Save Best Model State
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
            }, os.path.join(fold_checkpoint_dir, "best_model.pth"))

            # Logic: Move weights from 'current_epoch' to 'best_epoch' directory
            current_weights_dir = f"{fold_attn_dir}/current_epoch"
            best_weights_dir = f"{fold_attn_dir}/best_epoch"
            
            # Clean old best weights
            if os.path.exists(best_weights_dir):
                import shutil
                shutil.rmtree(best_weights_dir)
            
            # Promote current weights to best
            if os.path.exists(current_weights_dir):
                os.rename(current_weights_dir, best_weights_dir)
                logging.info(f"☑︎ Weights saved for Fold {fold}, Epoch {epoch+1} (RMSE: {val_rmse:.4f})")

        else:
            patience_counter += 1
            
            # Delete current epoch weights
            current_weights_dir = f"{fold_attn_dir}/current_epoch"
            if os.path.exists(current_weights_dir):
                import shutil
                shutil.rmtree(current_weights_dir)
                logging.info(f"[-] Weights deleted for Fold {fold}, Epoch {epoch+1} (not best)")
            
            logging.info(f"[!] No improvement in validation RMSE. Patience: {patience_counter}/{training_config['early_stopping_patience']}")
            if patience_counter >= training_config['early_stopping_patience']:
                logging.info("[STOP] Early stopping triggered. Training terminated.")
                break

        plot_statics(FILE_NAME, f"Fold {fold}", train_rmses, val_rmses)

    if fold != SAVE_FOLD_NUMBER:  # Only keep the specified fold
        current_fold_weights_dir = f"{WEIGHTS_DIR}/{FILE_NAME}/fold_{fold}"
        
        if os.path.exists(current_fold_weights_dir):
            shutil.rmtree(current_fold_weights_dir)
            logging.info(f"[-] Deleted weights for Fold {fold} (not Fold {SAVE_FOLD_NUMBER})")
        
    else:
        logging.info(f"☑︎ Keeping Fold {fold} as the fixed fold to save")

# Save the specified fold's results
target_fold_result_dir = f"{RESULTS_DIR}/{FILE_NAME}"
target_fold_weights_dir = f"{WEIGHTS_DIR}/{FILE_NAME}/fold_{SAVE_FOLD_NUMBER}"
os.makedirs(target_fold_result_dir, exist_ok=True)
os.makedirs(target_fold_weights_dir, exist_ok=True)

if os.path.exists(target_fold_weights_dir):
    # Save the target fold's predictions
    target_fold_data = torch.load(f'{data_config["cross_validation_data_dir"]}/cross_valid_fold_{SAVE_FOLD_NUMBER}.pt')
    target_fold_train_dataset = DrugResponseDataset(**target_fold_data['train'])
    target_fold_val_dataset = DrugResponseDataset(**target_fold_data['validation'])
    target_fold_train_loader = DataLoader(target_fold_train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    target_fold_val_loader = DataLoader(target_fold_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Load the best model
    target_fold_checkpoint = torch.load(f"{chpt_dir}/fold_{SAVE_FOLD_NUMBER}/best_model.pth")
    
    # Recreate model and load weights
    model = DrugResponseModel(
        pathway_gene_indices=pathway_gene_indices,
        gene_ffn_output_dim=GENE_FFN_OUTPUT_DIM,
        drug_ffn_output_dim=DRUG_FFN_OUTPUT_DIM,
        cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
        final_dim=FINAL_EMBEDDING_DIM,
        max_gene_slots=MAX_GENE_SLOTS,
        gene_input_dim=GENE_INPUT_DIM,
        drug_input_dim=DRUG_INPUT_DIM,
        isDiffer=IS_DIFFER,
        gene_ffn_hidden_dim=model_config['gene_ffn_hidden_dim'],
        drug_ffn_hidden_dim=model_config['drug_ffn_hidden_dim'],
        gene_ffn_dropout=model_config['gene_ffn_dropout'],
        drug_ffn_dropout=model_config['drug_ffn_dropout'],
        num_heads=model_config['num_heads'],
        depth=model_config['depth'],
        mlp_dropout=model_config['mlp_dropout'],
        final_dim_reduction_factor=model_config['final_dim_reduction_factor'],
        # DeepRegressorHead parameters
        **regressor_config
    ).to(DEVICE)
    model.load_state_dict(target_fold_checkpoint['model_state_dict'])
    
    model.eval()
    
    # Generate Predictions for Training Set
    target_fold_train_actuals, target_fold_train_predictions, target_fold_train_samples = [], [], []
    with torch.no_grad():
        for batch in tqdm(target_fold_train_loader, desc=f"Generating Fold {SAVE_FOLD_NUMBER} train predictions"):
            outputs, loss, rmse, sample_indices, labels = process_batch(0, batch, 0, model, criterion, DEVICE, mode="val", is_best_epoch=False, save_weights=False)
            target_fold_train_actuals.extend(labels)
            target_fold_train_predictions.extend(outputs)
            target_fold_train_samples.extend(sample_indices)
    
    # Generate Predictions for Validation Set
    target_fold_val_actuals, target_fold_val_predictions, target_fold_val_samples = [], [], []
    with torch.no_grad():
        for batch in tqdm(target_fold_val_loader, desc=f"Generating Fold {SAVE_FOLD_NUMBER} validation predictions"):
            outputs, loss, rmse, sample_indices, labels = process_batch(0, batch, 0, model, criterion, DEVICE, mode="val", is_best_epoch=False, save_weights=False)
            target_fold_val_actuals.extend(labels)
            target_fold_val_predictions.extend(outputs)
            target_fold_val_samples.extend(sample_indices)

    # Save the target fold validation predictions
    torch.save({
        "actuals": target_fold_train_actuals,
        "predictions": target_fold_train_predictions,
        "sample_indices": target_fold_train_samples
    }, os.path.join(target_fold_result_dir, f"train_results_fold_{SAVE_FOLD_NUMBER}.pt"))
    
    # Save the target fold validation predictions
    torch.save({
        "actuals": target_fold_val_actuals,
        "predictions": target_fold_val_predictions,
        "sample_indices": target_fold_val_samples
    }, os.path.join(target_fold_result_dir, f"val_results_fold_{SAVE_FOLD_NUMBER}.pt"))
    
    logging.info(f"☑︎ Fold {SAVE_FOLD_NUMBER} train and validation results saved in {target_fold_result_dir}")
    
    # Finalize Weight Directory Structure
    best_epoch_weights = f"{target_fold_weights_dir}/best_epoch"
    if os.path.exists(best_epoch_weights):
        for file in os.listdir(best_epoch_weights):
            src = os.path.join(best_epoch_weights, file)
            dst = os.path.join(target_fold_weights_dir, file)
            shutil.copy2(src, dst)
        logging.info(f"☑︎ Fold {SAVE_FOLD_NUMBER} weights saved in {target_fold_weights_dir}")
    
    # Cleanup temporary directories
    if os.path.exists(best_epoch_weights):
        shutil.rmtree(best_epoch_weights)
    current_epoch_dir = f"{target_fold_weights_dir}/current_epoch"
    if os.path.exists(current_epoch_dir):
        shutil.rmtree(current_epoch_dir)
else:
    logging.warning(f"[!] Fold {SAVE_FOLD_NUMBER} weights directory not found!")
