# inference.py
import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import LLM_CONFIGS, TRAINING_CONFIG
from data_pipeline import get_dataloaders
from model import TimeLLM


def main():
    parser = argparse.ArgumentParser(description="Inference for Time-LLM on stock time series data.")
    parser.add_argument("--profile", type=str, default="default",
                        choices=list(LLM_CONFIGS.keys()),
                        help="LLM configuration profile from config.py")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to model checkpoint (state_dict .pt file)")
    parser.add_argument("--out_csv", type=str, default="predictions.csv",
                        help="Output CSV file for predictions")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    args = parser.parse_args()

    # Load LLM settings
    profile_cfg = LLM_CONFIGS[args.profile]
    prompt_llm = profile_cfg["prompt_llm"]
    backbone_llm = profile_cfg["backbone_llm"]

    # Training hyperparameters
    input_len = TRAINING_CONFIG["input_len"]
    pred_len = TRAINING_CONFIG["pred_len"]
    batch_size = TRAINING_CONFIG["batch_size"]

    # Prepare DataLoader (mode=test returns entire series)
    feature_cols = None  # use default columns defined in data_pipeline or pass via config
    test_loader, (scaler_X, scaler_y) = get_dataloaders(
        feature_cols=feature_cols,
        in_len=input_len,
        out_len=pred_len,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeLLM(
        prompt_llm=prompt_llm,
        backbone_llm=backbone_llm,
        num_series=test_loader.dataset.X.shape[1],
        patch_len=16,
        patch_stride=16,
        patch_dim=16,
        model_dim=512,
        num_prototypes=1000,
        num_heads=8,
        pred_len=pred_len
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    model.eval()

    # Inference loop
    all_preds = []
    all_indices = []
    with torch.no_grad():
        for idx, (xb, yb, y0) in enumerate(test_loader):
            # xb: (B, in_len, N), need (B, N, T)
            xb = xb.permute(0,2,1).to(device)
            # Build prompt strings
            domain_desc = "Stock hourly candles"
            task_inst = f"Predict next {pred_len} steps"
            stats = f"last_price {y0.mean().item():.3f}"
            # Model forward
            yhat = model(xb, domain_desc, task_inst, stats)
            # yhat: (B, N, pred_len)
            # select first series or aggregate if multi
            # reshape and inverse transform
            yhat_np = yhat.cpu().numpy()  # (B, N, pred_len)
            # For simplicity, assume single series
            yhat_flat = scaler_y.inverse_transform(yhat_np.reshape(-1,1)).flatten()
            all_preds.extend(yhat_flat.tolist())
            # Track start index for each prediction
            start_idx = idx * batch_size + input_len
            indices = list(range(start_idx, start_idx + pred_len))
            all_indices.extend(indices)

    # Build DataFrame and save
    df_out = pd.DataFrame({
        "index": all_indices,
        "prediction": all_preds
    })
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
