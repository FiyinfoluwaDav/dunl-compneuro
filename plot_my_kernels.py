import torch
import matplotlib.pyplot as plt
import os
import glob
import sys
sys.path.append('dunl')

def main():
    # 1. Automatically find the most recent experiment folder in 'results'
    results_dir = os.path.join("results")
    if not os.path.exists(results_dir):
        print("Results directory not found! Have you run the training yet?")
        return
        
    # Get all subdirectories in results/
    subfolders = [f.path for f in os.scandir(results_dir) if f.is_dir()]
    if not subfolders:
        print("No experiment folders found in results/")
        return
        
    # Sort to find the latest one based on modification time or name
    latest_experiment_path = max(subfolders, key=os.path.getmtime)
    print(f"Loading models from the latest experiment folder:\n{latest_experiment_path}\n")

    # 2. Define the paths to the models
    model_final_path = os.path.join(latest_experiment_path, 'model', 'model_final.pt')
    model_epoch99_path = os.path.join(latest_experiment_path, 'model', 'model_epoch99.pt')

    # Load model_final.pt
    print("Loading model_final.pt...")
    model_final = torch.load(model_final_path, map_location=torch.device('cpu'))
    if 'H' in model_final.state_dict():
        kernels_final = model_final.state_dict()['H'].detach().cpu().numpy()
        print(f"Shape of kernels from model_final.pt: {kernels_final.shape}")
    else:
        print("Could not find 'H' in model_final.pt")
        return

    # Load model_epoch99.pt
    print("Loading model_epoch99.pt...")
    model_epoch99 = torch.load(model_epoch99_path, map_location=torch.device('cpu'))
    if 'H' in model_epoch99.state_dict():
        kernels_epoch99 = model_epoch99.state_dict()['H'].detach().cpu().numpy()
        print(f"Shape of kernels from model_epoch99.pt: {kernels_epoch99.shape}")
    else:
        print("Could not find 'H' in model_epoch99.pt")
        return

    # 3. Plot the kernels side-by-side for comparison
    plt.figure(figsize=(15, 7))

    # Plot kernels from model_epoch99.pt
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    for i in range(kernels_epoch99.shape[0]):
        plt.plot(kernels_epoch99[i, 0, :], label=f'Kernel {i+1}')
    plt.title('Learned Kernels (model_epoch99.pt)')
    plt.xlabel('Time Bins')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot kernels from model_final.pt
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    for i in range(kernels_final.shape[0]):
        plt.plot(kernels_final[i, 0, :], label=f'Kernel {i+1}')
    plt.title('Learned Kernels (model_final.pt)')
    plt.xlabel('Time Bins')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save the figure so you have a permanent record of the plot
    output_img_path = os.path.join(latest_experiment_path, "kernel_comparison.png")
    plt.savefig(output_img_path)
    print(f"\nPlot saved successfully to: {output_img_path}")
    
    # Display the plot window
    plt.show()

if __name__ == "__main__":
    main()
