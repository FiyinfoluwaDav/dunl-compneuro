import requests
import json
import os

API_BASE = "http://127.0.0.1:8000/api"
OUT_DIR = "dashboard/frontend/public/demo_data"

os.makedirs(OUT_DIR, exist_ok=True)

def save_json(path, data):
    with open(os.path.join(OUT_DIR, path), "w") as f:
        json.dump(data, f)

print("Fetching experiments...")
exps = requests.get(f"{API_BASE}/experiments").json()
save_json("experiments.json", exps)

if exps:
    exp_id = exps[0]['id']
    print(f"Exporting data for experiment: {exp_id}")
    
    # Config
    config = requests.get(f"{API_BASE}/experiments/{exp_id}/config").json()
    save_json("config.json", config)
    
    # Checkpoints
    checkpoints = requests.get(f"{API_BASE}/experiments/{exp_id}/checkpoints").json()
    save_json("checkpoints.json", checkpoints)
    
    # Loss
    loss = requests.get(f"{API_BASE}/experiments/{exp_id}/loss").json()
    save_json("loss.json", loss)
    
    # Pick latest checkpoint
    cp = checkpoints[-1] if isinstance(checkpoints, list) and len(checkpoints) > 0 else 'model_final.pt'
    
    # Kernels
    kernels = requests.get(f"{API_BASE}/experiments/{exp_id}/kernels?checkpoint={cp}").json()
    save_json("kernels.json", kernels)
    
    # Reconstruction
    print("Exporting trial 0...")
    recon_0 = requests.get(f"{API_BASE}/experiments/{exp_id}/reconstruction?checkpoint={cp}&trial_idx=0").json()
    save_json("reconstruction_0.json", recon_0)
    
    num_trials = recon_0.get("num_trials", 1)
    for t in range(1, num_trials):
        print(f"Exporting trial {t}...")
        recon_t = requests.get(f"{API_BASE}/experiments/{exp_id}/reconstruction?checkpoint={cp}&trial_idx={t}").json()
        save_json(f"reconstruction_{t}.json", recon_t)

print("Demo data exported successfully!")
