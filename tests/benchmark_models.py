"""Benchmark parameter counts and inference times by loading actual checkpoints."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
import numpy as np
import yaml


RUNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs', 'train', 'runs')

# Map run folder -> display name
RUNS = [
    ("MLP",           "MLP"),
    ("MAMBA2_6",      "MAMBA2"),
    ("MAMBA2_RPMG_6", "MAMBA2 + RPMG"),
    ("TF",            "TRANSFORMER"),
    ("TF_RMGP",       "TRANSFORMER + RPMG"),
]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(model, batch, gt, device, n_warmup=50, n_runs=200):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(batch, gt)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch, gt)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return np.mean(times), np.std(times)


def load_net_from_checkpoint(run_dir, device):
    """Load the inner net from a Lightning checkpoint using its hydra config."""
    config_path = os.path.join(run_dir, '.hydra', 'config.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    net_cfg = cfg['model']['net']
    target = net_cfg.pop('_target_')

    # Import the class from target string
    module_path, class_name = target.rsplit('.', 1)
    import importlib
    mod = importlib.import_module(module_path)
    NetClass = getattr(mod, class_name)

    # Instantiate fresh model with exact config
    net = NetClass(**net_cfg).to(device)

    # Find best checkpoint (not last.ckpt)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        # MLP has a different checkpoint path
        ckpt_dir = os.path.join(run_dir, 'tensorboard', 'version_0', 'checkpoints')

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt') and f != 'last.ckpt']
    if not ckpt_files:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])

    # Load state dict from Lightning checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['state_dict']

    # Strip "net." prefix from Lightning module keys
    net_state = {k.replace('net.', '', 1): v for k, v in state_dict.items() if k.startswith('net.')}
    net.load_state_dict(net_state)

    return net, ckpt_path


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    batch_size = 1
    seq_len = 11
    input_dim = 768

    dummy_features = torch.randn(batch_size, seq_len, input_dim, device=device)
    dummy_rot = torch.zeros(batch_size, seq_len, device=device)
    dummy_weight = torch.ones(batch_size, device=device)
    dummy_gt = torch.randn(batch_size, seq_len, 6, device=device)
    batch = (dummy_features, dummy_rot, dummy_weight)

    # Warmup CUDA with a dummy operation to avoid cold-start bias
    if device.type == 'cuda':
        _warmup = torch.randn(64, 768, device=device) @ torch.randn(768, 768, device=device)
        torch.cuda.synchronize()
        del _warmup

    results = []

    for run_folder, display_name in RUNS:
        run_dir = os.path.join(RUNS_DIR, run_folder)
        print(f"Loading {display_name} from {run_folder}...", end=" ")
        net, ckpt_path = load_net_from_checkpoint(run_dir, device)
        n_params = count_parameters(net)
        mean_t, std_t = benchmark_inference(net, batch, dummy_gt, device)
        results.append((display_name, n_params, mean_t, std_t))
        print(f"OK ({n_params:,} params, {mean_t:.3f} ms)")
        del net
        torch.cuda.empty_cache()

    # Print results table
    print(f"\n{'Model':<22} {'Params':>10} {'Params (K)':>10} {'Time (ms)':>12} {'Std (ms)':>10}")
    print("-" * 68)
    for name, params, mean_t, std_t in results:
        print(f"{name:<22} {params:>10,} {params/1000:>10.1f} {mean_t:>12.3f} {std_t:>10.3f}")

    # LaTeX table
    print("\n\nLaTeX table:")
    print(r"\begin{tabular}{l r r r}")
    print(r"\toprule")
    print(r"Model & Params & Params (K) & Inference (ms) \\")
    print(r"\midrule")
    for name, params, mean_t, std_t in results:
        print(f"{name} & {params:,} & {params/1000:.1f} & {mean_t:.3f} $\\pm$ {std_t:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
