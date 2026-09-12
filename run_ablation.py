"""
消融实验自动化脚本
分别训练并测试：Baseline, HFP-only, LEGM-only, MGDC-only, FFCM-only, All-modules
用法：
    python run_ablation.py [--skip-training]
    --skip-training: 跳过训练，只测试和评估（用于已有模型的情况）
"""

import os
import time
import argparse
import numpy as np
import torch
import csv

from args import args
import utils_enhanced as utils
from Models_enhanced import Generator_Enhanced
from generate_enhanced import generate
from evaluate import evaluate_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = [
    {"name": "Baseline",    "suffix": "baseline",    "config": {"use_hfp": False, "use_legm": False, "use_mgdc": False, "use_ffcm": False}, "trained": False},
    {"name": "HFP_only",    "suffix": "hfp_only",    "config": {"use_hfp": True,  "use_legm": False, "use_mgdc": False, "use_ffcm": False}, "trained": False},
    {"name": "LEGM_only",   "suffix": "legm_only",   "config": {"use_hfp": False, "use_legm": True,  "use_mgdc": False, "use_ffcm": False}, "trained": False},
    {"name": "MGDC_only",   "suffix": "mgdc_only",   "config": {"use_hfp": False, "use_legm": False, "use_mgdc": True,  "use_ffcm": False}, "trained": False},
    {"name": "FFCM_only",   "suffix": "ffcm_only",   "config": {"use_hfp": False, "use_legm": False, "use_mgdc": False, "use_ffcm": True},  "trained": False},
    {"name": "All_modules", "suffix": "all_modules", "config": {"use_hfp": True,  "use_legm": True,  "use_mgdc": True,  "use_ffcm": True},  "trained": False},
]

TEST_IR_DIR = "./MSRS/test/ir/"
TEST_VI_DIR = "./MSRS/test/vi/"


def find_final_model(suffix):
    """找到某个实验最新的 Final_G_Epoch_*.model"""
    import glob
    finals = sorted(glob.glob(f"./models_training_5_{suffix}/Final_G_Epoch_*.model"))
    return finals[-1] if finals else None


def train_model(cfg):
    import train_enhanced
    train_data_ir = utils.list_images(args.train_ir)
    train_data_vi = utils.list_images(args.train_vi)

    print(f"\n{'='*60}")
    print(f"Training: {cfg['name']}")
    print(f"Config: {cfg['config']}")
    print(f"{'='*60}\n")

    train_enhanced.train(
        train_data_ir, train_data_vi,
        use_hfp=cfg['config']['use_hfp'],
        use_legm=cfg['config']['use_legm'],
        use_mgdc=cfg['config']['use_mgdc'],
        use_ffcm=cfg['config']['use_ffcm'],
        save_suffix=cfg['suffix']
    )


def test_model(cfg):
    model_path = find_final_model(cfg['suffix'])
    result_dir = f"results_{cfg['suffix']}"

    if model_path is None:
        print(f"[WARN] Model not found for {cfg['name']}, skipping test.")
        return False

    os.makedirs(result_dir, exist_ok=True)
    for f in os.listdir(result_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(result_dir, f))

    model = Generator_Enhanced(**cfg['config'])
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    model.to(device)

    print(f"\nTesting: {cfg['name']} -> {result_dir}/")
    with torch.no_grad():
        begin = time.time()
        # MSRS：ir 与 vi 同名对应
        ir_files = sorted(f for f in os.listdir(TEST_IR_DIR) if f.endswith('.png'))
        for i, ir_name in enumerate(ir_files):
            index = i + 1
            ir_path = TEST_IR_DIR + ir_name
            vis_path = TEST_VI_DIR + ir_name
            generate(model, ir_path, vis_path, result_dir, index, mode='L')
        end = time.time()
        print(f"  Test time: {end - begin:.2f}s")
    return True


def evaluate_model(cfg):
    result_dir = f"results_{cfg['suffix']}"
    metrics_list = []

    # MSRS：ir 与 vi 同名对应
    ir_files = sorted(f for f in os.listdir(TEST_IR_DIR) if f.endswith('.png'))
    for i, ir_name in enumerate(ir_files, start=1):
        ir_path = os.path.join(TEST_IR_DIR, ir_name)
        vis_path = os.path.join(TEST_VI_DIR, ir_name)
        fus_name = f"100{i}.png" if i < 10 else f"10{i}.png"
        fus_path = os.path.join(result_dir, fus_name)

        if os.path.exists(fus_path):
            try:
                m = evaluate_pair(ir_path, vis_path, fus_path)
                metrics_list.append(m)
            except Exception as e:
                print(f"  [ERROR] Evaluating {fus_path}: {e}")

    if not metrics_list:
        print(f"[WARN] No valid results for {cfg['name']}")
        return None

    avg = {k: np.mean([r[k] for r in metrics_list]) for k in metrics_list[0].keys()}
    return {
        'name': cfg['name'],
        'EN': avg['EN'], 'MI': avg['MI'], 'SD': avg['SD'], 'SF': avg['SF'],
        'SSIM_ir': avg['SSIM_ir'], 'SSIM_vis': avg['SSIM_vis'], 'Qabf': avg['Qabf'],
    }


def print_summary(results):
    print("\n" + "=" * 95)
    print("消融实验结果汇总（MSRS 测试集 361 对）")
    print("=" * 95)
    header = f"{'Config':>14} | {'EN':>7} | {'MI':>7} | {'SD':>7} | {'SF':>7} | {'SSIM_ir':>8} | {'SSIM_vis':>8} | {'Qabf':>8}"
    print(header)
    print("-" * 95)

    for r in results:
        if r is None:
            continue
        print(f"{r['name']:>14} | {r['EN']:>7.4f} | {r['MI']:>7.4f} | {r['SD']:>7.2f} | "
              f"{r['SF']:>7.2f} | {r['SSIM_ir']:>8.4f} | {r['SSIM_vis']:>8.4f} | {r['Qabf']:>8.4f}")

    print("=" * 95)
    print("\n说明：EN/MI/SD/SF/SSIM/Qabf 越高越好")
    print("      测试集为 MSRS 361 对图像\n")


def main():
    parser = argparse.ArgumentParser(description='ICAFusion Ablation Study')
    parser.add_argument('--skip-training', action='store_true',
                        help='跳过训练，只测试和评估已有模型')
    parser.add_argument('--only', type=str, default=None,
                        help='只运行指定实验，如: HFP_only, MGDC_only')
    args_cli = parser.parse_args()

    experiments = EXPERIMENTS
    if args_cli.only:
        experiments = [e for e in EXPERIMENTS if e['name'] == args_cli.only]
        if not experiments:
            print(f"[ERROR] Unknown experiment: {args_cli.only}")
            print(f"Available: {[e['name'] for e in EXPERIMENTS]}")
            return

    results_summary = []

    for cfg in experiments:
        model_path = find_final_model(cfg['suffix'])

        # 1. 训练
        if not args_cli.skip_training and model_path is None:
            try:
                train_model(cfg)
                cfg['trained'] = True
            except Exception as e:
                print(f"[ERROR] Training {cfg['name']} failed: {e}")
                results_summary.append(None)
                continue
        else:
            if model_path is not None:
                print(f"\n[SKIP] Model already exists: {model_path}")
                cfg['trained'] = True

        # 2. 测试
        if cfg['trained']:
            success = test_model(cfg)
            if not success:
                results_summary.append(None)
                continue
        else:
            results_summary.append(None)
            continue

        # 3. 评估
        metrics = evaluate_model(cfg)
        results_summary.append(metrics)

        if metrics:
            print(f"  -> EN={metrics['EN']:.4f}, MI={metrics['MI']:.4f}, "
                  f"SD={metrics['SD']:.2f}, SF={metrics['SF']:.2f}")

    # 汇总
    print_summary(results_summary)

    # 保存CSV
    csv_path = "ablation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Config', 'EN', 'MI', 'SD', 'SF', 'SSIM_ir', 'SSIM_vis', 'Qabf'])
        for r in results_summary:
            if r:
                writer.writerow([r['name'], r['EN'], r['MI'], r['SD'], r['SF'],
                                 r['SSIM_ir'], r['SSIM_vis'], r['Qabf']])
    print(f"结果已保存到: {csv_path}")


if __name__ == '__main__':
    main()
