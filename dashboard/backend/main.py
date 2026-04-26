import os
import glob
import sys
import pickle
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure the 'dunl' folder is in path so torch can load the model classes
sys.path.append(os.path.join(os.getcwd(), 'dunl'))

app = FastAPI(title="DUNL Analytics API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.join(os.getcwd(), "results")

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

@app.get("/api/experiments/{exp_id}/kernels")
def get_experiment_kernels(exp_id: str):
    """Loads model_final.pt and returns the kernel weights as arrays."""
    model_path = os.path.join(RESULTS_DIR, exp_id, "model", "model_final.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
        
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = model.state_dict()
        
        if 'H' not in state_dict:
            raise HTTPException(status_code=404, detail="Kernels ('H') not found in model")
            
        # Shape is usually (num_kernels, 1, kernel_length)
        kernels_tensor = state_dict['H'].detach().cpu()
        kernels_list = kernels_tensor.numpy().tolist()
        
        return {
            "kernels": kernels_list,
            "shape": list(kernels_tensor.shape)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Add a simple health check
@app.get("/api/health")
def health_check():
    return {"status": "ok"}
