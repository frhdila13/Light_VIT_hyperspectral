import os
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchinfo import summary # Using torchinfo instead of torchsummaryX
from utils.dataset import load_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test
from utils.utils import Draw
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='gscvit')
    parser.add_argument("--dataset_name", type=str, default="chikusei") # Default to your set
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--img_name", type=str, default="subset_hyper_Chikusei.tif") # Add these
    parser.add_argument("--gt_name", type=str, default="subset_gt_Chikusei.tif")   # Add these
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=8) # Changed to 8 for GSCViT math
    parser.add_argument("--num_run", type=int, default=1)   # Default to 1 for testing
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--bs", type=int, default=32) 
    parser.add_argument("--ratio", type=float, default=0.8) # 0.8 total (Train+Val)

    opts = parser.parse_args()
    device = torch.device("cuda:{}".format(opts.device))

    # Calculate split percentages for printing
    train_per = opts.ratio * 0.75 # 0.8 * 0.75 = 0.6
    val_per = opts.ratio * 0.25   # 0.8 * 0.25 = 0.2
    test_per = 1 - opts.ratio     # 1 - 0.8 = 0.2

    print(f"Experiments on GPU: {opts.device}")
    print(f"Data Split: {train_per*100}% Train, {val_per*100}% Val, {test_per*100}% Test")

    # --- Load Data ---
    # Using the manual filenames from argparse
    image, gt, labels = load_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands = image.shape[-1]
    seeds = [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410]
    results = []

    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        print(f"\n--- Run {run + 1} / {opts.num_run} ---")

        # --- 60/20/20 Split Logic ---
        # Split 80% (TrainVal) and 20% (Test)
        trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])
        
        # Split that 80% into 75% of it (which is 60% total) and 25% of it (which is 20% total)
        train_gt, val_gt = sample_gt(trainval_gt, 0.75, seeds[run]) 
        del trainval_gt

        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)

        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=False, shuffle=False)

        # load model with classes and bands
        model = get_model(opts.model, opts.dataset_name, opts.patch_size, num_classes, num_bands)

        if run == 0:
            split_info_print(train_gt, val_gt, test_gt, labels)
            print("Network Information Summary:")
            # Updated torchinfo summary call
            summary(model, input_size=(1, 1, num_bands, opts.patch_size, opts.patch_size), device='cpu')

        model = model.to(device)
        optimizer, scheduler = load_scheduler(opts.model, model)
        criterion = nn.CrossEntropyLoss()

        model_dir = f"./checkpoints/{opts.model}/{opts.dataset_name}/{run}"

        try:
            train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('Training interrupted by user.')

        # test the model
        probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)
        prediction = np.argmax(probabilities, axis=-1)

        run_results = metrics(prediction, test_gt, n_classes=num_classes)
        results.append(run_results)
        show_results(run_results, label_values=labels)

        # Draw classification map
        Draw(model, image, gt, opts.patch_size, opts.dataset_name, opts.model, num_classes)
        
        # === ADDITIONAL INFERENCE CODE START ===
        print("\nStarting Full-Scene Inference (Classifying every pixel)...")
        model.eval()
        height, width, _ = image.shape
        full_prediction = np.zeros((height, width))
        ps = opts.patch_size // 2
        
        # Pad the image using the same reflect logic as your HSIDataset
        padded_img = np.pad(image, ((ps, ps), (ps, ps), (0, 0)), mode='reflect')
        
        # Sliding window inference
        with torch.no_grad():
            for i in range(height):
                patches = []
                for j in range(width):
                    # Extract patch centered at (i, j)
                    patch = padded_img[i:i+opts.patch_size, j:j+opts.patch_size, :]
                    patch = patch.transpose((2, 0, 1)) # to (C, H, W)
                    patches.append(patch)
                    
                    # Batch processing to maximize GPU efficiency
                    if len(patches) == 128 or j == width - 1:
                        batch = torch.from_numpy(np.array(patches)).float().to(device)
                        batch = batch.unsqueeze(1) # Add 4th dimension for GSCViT (1, C, H, W)
                        
                        output = model(batch)
                        pred = torch.argmax(output, dim=1)
                        
                        # Store prediction back into the map
                        start_j = j - len(patches) + 1
                        full_prediction[i, start_j:j+1] = pred.cpu().numpy()
                        patches = []
                
                if (i + 1) % 50 == 0:
                    print(f"Processed line {i+1}/{height}")

        # Use your existing DrawResult from utils.py to color the map
        # We use DrawResult(h, w, n_classes, labels)
        from utils.utils import DrawResult
        # Note: We add 1 to full_prediction because your palette logic is likely 1-indexed
        full_colored_map = DrawResult(height, width, num_classes, full_prediction.reshape(-1) + 1)
        
        # Save the full image
        import matplotlib.pyplot as plt
        full_map_path = f"results/FULL_SCENE_{opts.model}_{opts.dataset_name}.png"
        
        # Create the results folder if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Define paths for both PNG (visual) and TIF (data)
        full_map_path_png = f"results/FULL_SCENE_{opts.model}_{opts.dataset_name}.png"
        full_map_path_tif = f"results/FULL_SCENE_{opts.model}_{opts.dataset_name}.tif"

        # Save as PNG for a quick preview (uses the colored map)
        import matplotlib.pyplot as plt
        plt.imsave(full_map_path_png, full_colored_map)

        # Save as TIF (saves the raw prediction labels 0-12)
        import tifffile as tiff
        # We convert to uint8 or uint16 to keep the file size small
        tiff.imwrite(full_map_path_tif, full_prediction.astype(np.uint8))

        print(f"Full-scene PNG saved to: {full_map_path_png}")
        print(f"Full-scene TIF saved to: {full_map_path_tif}")
        print(f"Full-scene map successfully saved to {full_map_path}")
        # === ADDITIONAL INFERENCE CODE END ===

        del model, train_set, train_loader, val_set, val_loader

    if opts.num_run > 1:
        show_results(results, label_values=labels, agregated=True)
