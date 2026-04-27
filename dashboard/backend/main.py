import os
import glob
import sys
import pickle
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Resolve project root based on this file's location
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure the 'dunl' folder is in path so torch can load the model classes
sys.path.append(os.path.join(ROOT_DIR, 'dunl'))

app = FastAPI(title="DUNL Analytics API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")

@app.get("/api/experiments")
def get_experiments():
    """Returns a list of all experiment folders sorted by latest first."""
    if not os.path.exists(RESULTS_DIR):
        return []
    
    subfolders = [f.path for f in os.scandir(RESULTS_DIR) if f.is_dir()]
    subfolders.sort(key=os.path.getmtime, reverse=True)
    
    experiments = []
    for folder in subfolders:
        exp_id = os.path.basename(folder)
        experiments.append({
            "id": exp_id,
            "name": exp_id,
            "path": folder
        })
    return experiments

@app.get("/api/experiments/{exp_id}/config")
def get_experiment_config(exp_id: str):
    """Loads the params.pickle for a specific experiment."""
    params_path = os.path.join(RESULTS_DIR, exp_id, "params.pickle")
    if not os.path.exists(params_path):
        raise HTTPException(status_code=404, detail="Config not found")
        
    with open(params_path, "rb") as f:
        # Some objects in pickle might not be JSON serializable, so we clean them
        raw_params = pickle.load(f)
        
    clean_params = {}
    for k, v in raw_params.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            clean_params[k] = v
        else:
            clean_params[k] = str(v)
            
    return clean_params

@app.get("/api/experiments/{exp_id}/loss")
def get_experiment_loss(exp_id: str):
    """Parses the Tensorboard log and returns the training loss curve."""
    exp_dir = os.path.join(RESULTS_DIR, exp_id)
    if not os.path.exists(exp_dir):
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    # Find the tfevents file
    tfevent_files = [f.path for f in os.scandir(exp_dir) if f.is_file() and 'tfevents' in f.name]
    if not tfevent_files:
        raise HTTPException(status_code=404, detail="Tensorboard logs not found")
        
    tfevent_file = tfevent_files[0]
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        # Limit the size so it doesn't consume huge memory
        ea = EventAccumulator(tfevent_file, size_guidance={'scalars': 1000})
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        
        data = {}
        for tag in tags:
            events = ea.Scalars(tag)
            data[tag] = [{"step": e.step, "value": e.value} for e in events]
            
        return data
    except ImportError:
        raise HTTPException(status_code=500, detail="Tensorboard package is not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse tensorboard logs: {str(e)}")

@app.get("/api/experiments/{exp_id}/checkpoints")
def get_experiment_checkpoints(exp_id: str):
    """Returns a list of all model checkpoints (.pt files) for an experiment."""
    model_dir = os.path.join(RESULTS_DIR, exp_id, "model")
    if not os.path.exists(model_dir):
        return []
    
    files = [f.name for f in os.scandir(model_dir) if f.is_file() and f.name.endswith(".pt")]
    # Try to sort logically (epoch1.pt, epoch2.pt... final.pt)
    def sort_key(name):
        if "final" in name:
            return float('inf')
        # Extract number if possible
        import re
        match = re.search(r'\d+', name)
        return int(match.group()) if match else 0
        
    files.sort(key=sort_key)
    return files

@app.get("/api/experiments/{exp_id}/kernels")
def get_experiment_kernels(exp_id: str, checkpoint: str = "model_final.pt"):
    """Loads a specific model checkpoint and returns the kernel weights as arrays."""
    # Prevent directory traversal attacks
    checkpoint = os.path.basename(checkpoint)
    model_path = os.path.join(RESULTS_DIR, exp_id, "model", checkpoint)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint} not found")
        
    try:
        # Pass weights_only=False to allow loading the custom model classes
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        state_dict = model.state_dict()
        
        if 'H' not in state_dict:
            raise HTTPException(status_code=404, detail="Kernels ('H') not found in model")
            
        # Shape is usually (num_kernels, 1, kernel_length)
        kernels_tensor = state_dict['H'].detach().cpu()
        kernels_list = kernels_tensor.numpy().tolist()
        
        # Compute Kernel Similarity Matrix (Cross-correlation)
        H_sq = kernels_tensor.squeeze(1) # (K, T)
        H_norm = torch.nn.functional.normalize(H_sq, p=2, dim=1)
        sim_matrix = torch.matmul(H_norm, H_norm.T).numpy().tolist()
        
        return {
            "kernels": kernels_list,
            "shape": list(kernels_tensor.shape),
            "similarity_matrix": sim_matrix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/api/experiments/{exp_id}/reconstruction")
def get_experiment_reconstruction(exp_id: str, checkpoint: str = "model_final.pt", trial_idx: int = 0):
    """Loads model checkpoint and dataset, runs inference, returns data for Heatmaps and Sparse Codes."""
    checkpoint = os.path.basename(checkpoint)
    model_path = os.path.join(RESULTS_DIR, exp_id, "model", checkpoint)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint} not found")
        
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        
        params_path = os.path.join(RESULTS_DIR, exp_id, "params.pickle")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
            
        data_paths = params.get("data_path", [])
        if not data_paths:
            raise Exception("No data_path found in params")
            
        rel_data_path = data_paths[0].replace("../", "")
        abs_data_path = os.path.normpath(os.path.join(ROOT_DIR, rel_data_path))
        
        if not os.path.exists(abs_data_path):
            raise Exception(f"Data file not found at {abs_data_path}")
            
        data = torch.load(abs_data_path, map_location=torch.device('cpu'), weights_only=False)
        
        y_all = data['y']
        a_all = data['a']
        
        # We will use the requested trial for visualization
        y_load = y_all[[trial_idx]].float()
        a_load = a_all[[trial_idx]].float()
        
        # Reshape to (batch*neurons, 1, time) as expected by the model when sharing kernels
        y = torch.reshape(y_load, (y_load.shape[0] * y_load.shape[1], 1, y_load.shape[2]))
        a = torch.reshape(a_load, (a_load.shape[0] * a_load.shape[1], 1, a_load.shape[2]))
        
        with torch.no_grad():
            x_est, a_est = model.encode(y, a)
            yhat = model.decode(x_est, a_est)
            
            # Decompose by kernel
            num_kernels = x_est.shape[1]
            components = []
            for k in range(num_kernels):
                x_k = torch.zeros_like(x_est)
                x_k[:, k, :] = x_est[:, k, :]
                yhat_k = model.decode(x_k, torch.zeros_like(a_est))
                yhat_k_reshaped = torch.reshape(yhat_k, (y_load.shape[0], y_load.shape[1], y_load.shape[2]))
                # Average across neurons to get a 1D trace for the whole trial
                comp_trace = yhat_k_reshaped[0].mean(dim=0).tolist()
                components.append(comp_trace)
            
        yhat_reshaped = torch.reshape(yhat, (y_load.shape[0], y_load.shape[1], y_load.shape[2]))
        
        model_distribution = params.get("model_distribution", "gaussian")
        if model_distribution == "poisson":
            rate = torch.exp(yhat_reshaped)
        elif model_distribution == "binomial":
            rate = torch.sigmoid(yhat_reshaped)
        else:
            rate = yhat_reshaped
            
        residuals = (y_load - rate)[0].tolist()
        
        # Calculate Quantitative Metrics
        ss_res = torch.sum((y_load - rate) ** 2).item()
        ss_tot = torch.sum((y_load - y_load.mean()) ** 2).item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        if model_distribution == "poisson":
            nll = torch.sum(rate - y_load * torch.log(rate + 1e-8)).item()
        elif model_distribution == "binomial":
            nll = -torch.sum(y_load * torch.log(rate + 1e-8) + (1 - y_load) * torch.log(1 - rate + 1e-8)).item()
        else:
            nll = 0.5 * ss_res
            
        metrics = {
            "r2": round(r2, 4),
            "nll": round(nll, 4)
        }
        
        # Aggregate non-zero codes across all neurons for the trial-level scatter plot
        x_trial = x_est.abs().sum(dim=0).tolist()
            
        return {
            "y": y_load[0].tolist(),
            "rate": rate[0].tolist(),
            "residuals": residuals,
            "components": components,
            "x": x_trial,
            "num_trials": y_all.shape[0],
            "metrics": metrics
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed reconstruction: {str(e)}")

# Add a simple health check
@app.get("/api/health")
def health_check():
    return {"status": "ok"}
