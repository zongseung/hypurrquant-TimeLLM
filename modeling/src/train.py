# train.py
import argparse
import torch
import torch.nn.functional as F
from data_pipeline import get_dataloaders
from model import TimeLLM
from config import TRAINING_CONFIG, LLM_CONFIGS, USE_DEEPSPEED, USE_ACCELERATE
from utils import set_seed, get_logger, save_checkpoint

def main():
    # ─── 1) CLI & Config ────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="default",
                        choices=list(LLM_CONFIGS.keys()),
                        help="LLM 설정 프로파일 이름")
    parser.add_argument("--data_path", type=str, required=False,
                        help="(Optional) DB 대신 CSV 등 쓰실 경우")
    args = parser.parse_args()

    # 시드 고정, 로거 준비
    set_seed(TRAINING_CONFIG["seed"])
    logger = get_logger("logs")

    # LLM 프로파일 로드
    profile_cfg   = LLM_CONFIGS[args.profile]
    prompt_llm    = profile_cfg["prompt_llm"]
    backbone_llm  = profile_cfg["backbone_llm"]

    # 학습 하이퍼파라미터
    feature_cols = ["Open","High","Low","Close","Volume","total_transaction"]
    close_idx = feature_cols.index("Close")
    in_len       = TRAINING_CONFIG["input_len"]
    out_len      = TRAINING_CONFIG["pred_len"]
    batch_size   = TRAINING_CONFIG["batch_size"]
    epochs       = TRAINING_CONFIG["epochs"]
    lr           = TRAINING_CONFIG["lr"]

    # ─── 2) DataLoader ──────────────────────────────────────────────
    train_loader, val_loader, test_loader, (scaler_X, scaler_y) = get_dataloaders(
        feature_cols=feature_cols,
        in_len=in_len,
        out_len=out_len,
        batch_size=batch_size,
    )

    # ─── 3) 모델 & 옵티마이저 초기화 ─────────────────────────────────
    model = TimeLLM(
        prompt_llm=prompt_llm,
        backbone_llm=backbone_llm,
        num_series=len(feature_cols),
        patch_len=16,
        patch_stride=16,
        patch_dim=16,
        model_dim=768,
        num_prototypes=1000,
        num_heads=8,
        pred_len=out_len
    ).cuda()

    optim = torch.optim.Adam(
        list(model.reprog.parameters()) +
        list(model.output_proj.parameters()) +
        list(model.prompt.parameters()),
        lr=lr
    )

    # (옵션) DeepSpeed / Accelerate 연동
    if USE_DEEPSPEED:
        import deepspeed
        model, optim, _, _ = deepspeed.initialize(
            args=None, model=model, optimizer=optim,
            config="configs/ds_config.json"
        )
    if USE_ACCELERATE:
        from accelerate import Accelerator
        accelerator = Accelerator()
        model, optim, train_loader = accelerator.prepare(
            model, optim, train_loader
        )

    # ─── 4) 학습 루프 ───────────────────────────────────────────────
    for epoch in range(1, epochs+1):
        # Train
        model.train()
        total_train_loss = 0.0
        for xb, yb, y0 in train_loader:
            xb = xb.cuda()
            yb = yb.cuda()

            optim.zero_grad()
            yhat = model(
                xb,
                domain_desc="Stock hourly candles",
                task_inst=f"Predict next {out_len} steps",
                stats=f"last_price {y0.mean().item():.3f}"
            )
            loss = F.mse_loss(yhat[:, close_idx, :], yb.squeeze(1))

            if USE_ACCELERATE:
                accelerator.backward(loss)
            else:
                loss.backward()
            optim.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, y0 in val_loader:
                xb = xb.cuda()
                yb = yb.cuda()
                yhat = model(
                    xb,
                    domain_desc="Stock hourly candles",
                    task_inst=f"Predict next {out_len} steps",
                    stats=f"last_price {y0.mean().item():.3f}"
                )
                loss = F.mse_loss(yhat[:, close_idx, :], yb.squeeze(1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Test
        total_test_loss = 0.0
        with torch.no_grad():
            for xb, yb, y0 in test_loader:
                xb = xb.cuda()
                yb = yb.cuda()
                yhat = model(
                    xb,
                    domain_desc="Stock hourly candles",
                    task_inst=f"Predict next {out_len} steps",
                    stats=f"last_price {y0.mean().item():.3f}"
                )
                loss = F.mse_loss(yhat[:, close_idx, :], yb.squeeze(1))
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        if epoch % 10 == 0:
            logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # 체크포인트 저장
        save_checkpoint(model, optim, epoch, ckpt_dir="ckpts")

    logger.info("Training completed.")

if __name__ == "__main__":
    main()
