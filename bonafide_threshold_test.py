import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Project-specific imports
from config.config import get_cfg_defaults
from data.make_dataset import make_data
from data.get_true_dataset import get_true
from models.OneClass.model_lit import CA_SOADD_Lit

warnings.filterwarnings("ignore")

# ==========================================
# 1. Metric Utilities
# ==========================================

def compute_eer(y_true, y_score):
    """
    Computes EER and AUC.
    y_true: 0 for negative, 1 for positive.
    y_score: similarity scores (higher indicates more likely to be bonafide).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        fnr = 1.0 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = (fpr[min_index] + fnr[min_index]) / 2.0
    return eer, roc_auc


# ==========================================
# 2. Threshold Calibration Functions
# ==========================================

def _get_centroid(model, feat_device):
    """
    Retrieves the centroid from different potential Lightning model locations.
    """
    if hasattr(model, "loss_fn") and hasattr(model.loss_fn, "centroid") and model.loss_fn.centroid is not None:
        centroid = model.loss_fn.centroid
    elif hasattr(model, "centroid"):
        centroid = model.centroid
    else:
        raise RuntimeError("Centroid not found in model (check model.loss_fn.centroid or model.centroid).")
    return centroid.to(feat_device)


def collect_cosine_sim(model, loader, device, only_bonafide=False, max_batches=None):
    """
    Extracts cosine-to-centroid similarity scores from a dataloader.
    Returns:
        sims: ndarray of cosine similarities [-1, 1].
        labels: ndarray of ground truth (1=bonafide, 0=spoof).
    """
    model.eval()
    all_sims = []
    all_labels = []

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="Extracting scores")):
            if max_batches is not None and bi >= max_batches:
                break

            audio = batch["audio"].to(device)
            label = batch["label"].cpu().numpy()

            if only_bonafide:
                mask = (label == 1)
                if mask.sum() == 0:
                    continue
                audio = audio[mask]
                label = label[mask]

            if len(audio.shape) == 3:
                audio = audio[:, 0, :]

            # Forward pass through the backbone
            res = model.model(audio)
            feat = F.normalize(res["final_feat"], p=2, dim=1)

            # Similarity calculation
            centroid = _get_centroid(model, feat.device)
            centroid_norm = F.normalize(centroid, p=2, dim=0)

            sim = torch.matmul(feat, centroid_norm).detach().cpu().numpy()
            all_sims.append(sim)
            all_labels.append(label)

    if not all_sims:
        return np.array([]), np.array([])

    return np.concatenate(all_sims), np.concatenate(all_labels)


def calibrate_tau_from_bonafide(model, dev_loader, device, alphas, max_batches=None):
    """
    Calibrates decision thresholds (tau) using only bonafide dev data.
    tau_alpha is defined as the alpha-quantile of bonafide similarities.
    """
    sims, _ = collect_cosine_sim(
        model=model,
        loader=dev_loader,
        device=device,
        only_bonafide=True,
        max_batches=max_batches
    )
    if sims.size == 0:
        raise RuntimeError("No bonafide samples found for calibration.")

    # Calculate thresholds for each alpha (Target False Rejection Rates)
    return {a: float(np.quantile(sims, a)) for a in alphas}


def eval_at_thresholds(model, test_loader, device, taus, max_batches=None):
    """
    Evaluates FNR (FRR) and FPR (FAR) at fixed calibrated thresholds.
    Acceptance rule: similarity >= tau.
    """
    sims, labels = collect_cosine_sim(
        model=model,
        loader=test_loader,
        device=device,
        only_bonafide=False,
        max_batches=max_batches
    )
    
    bon = sims[labels == 1]
    spf = sims[labels == 0]

    results = {}
    for a, tau in taus.items():
        fnr = float(np.mean(bon < tau)) if bon.size > 0 else np.nan
        fpr = float(np.mean(spf >= tau)) if spf.size > 0 else np.nan
        results[a] = {"tau": tau, "FNR": fnr, "FPR": fpr}
    return results


def print_results_table(results, title=None):
    """
    Formats and prints the evaluation metrics table.
    """
    if title:
        print(f"\n{'='*80}\n{title}\n{'='*80}")

    print(f"{'alpha':>8} | {'tau (threshold)':>14} | {'FNR(%)':>8} | {'FPR(%)':>8}")
    print("-" * 55)
    for a in sorted(results.keys()):
        data = results[a]
        print(f"{a:>8.4f} | {data['tau']:>14.6f} | {data['FNR']*100:>8.3f} | {data['FPR']*100:>8.3f}")


# ==========================================
# 3. Main Execution
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Class Threshold Calibration")
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-ckpt", "--checkpoint", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--max_calib_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    args = parser.parse_args()

    # Load configuration
    cfg = get_cfg_defaults(f"config/experiments/{args.cfg}.yaml")
    cfg.DATASET.batch_size = args.batch_size
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')

    # Load Model
    print(f"Loading model: {args.checkpoint}")
    model = CA_SOADD_Lit()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device).eval()

    # Load Test Data
    _, dl = make_data(cfg.DATASET, args=args)
    test_loaders = dl.test if isinstance(dl.test, list) else [dl.test]

    # Load Calibration Data (Development set - Bonafide only)
    # Using get_true to fetch specific bonafide-only subsets
    _, calib_loader = get_true(["ASV5"], bs=16, label="val")

    # Target False Rejection Rates (FRR) for calibration
    target_alphas = [0.10, 0.05, 0.01, 0.005, 0.001]

    print("\nStarting bonafide-only calibration...")
    calibrated_taus = calibrate_tau_from_bonafide(
        model=model,
        dev_loader=calib_loader,
        device=device,
        alphas=target_alphas,
        max_batches=args.max_calib_batches
    )

    print("\nCalibrated Thresholds:")
    for a in sorted(calibrated_taus.keys()):
        print(f"  Target FNR {a*100:.1f}% -> Threshold (tau): {calibrated_taus[a]:.6f}")

    # Final Evaluation
    for i, loader in enumerate(test_loaders):
        print(f"\nEvaluating Test Set {i}...")
        results = eval_at_thresholds(
            model=model,
            test_loader=loader,
            device=device,
            taus=calibrated_taus,
            max_batches=args.max_test_batches
        )
        print_results_table(results, title=f"Test Set {i} Analysis")