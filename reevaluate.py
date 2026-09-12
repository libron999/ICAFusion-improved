"""快速重新评估所有消融实验结果（使用修复后的 Qabf）"""
import os
import numpy as np
from evaluate import evaluate_pair

EXPERIMENTS = [
    "baseline", "hfp_only", "legm_only", "mgdc_only", "ffcm_only", "all_modules"
]
NAMES = ["Baseline", "HFP_only", "LEGM_only", "MGDC_only", "FFCM_only", "All_modules"]
TEST_IR_DIR = "./MSRS/test/ir/"
TEST_VI_DIR = "./MSRS/test/vi/"

print("=" * 95)
print("消融实验结果汇总（Qabf 已修复）")
print("=" * 95)
header = f"{'Config':>14} | {'EN':>7} | {'MI':>7} | {'SD':>7} | {'SF':>7} | {'SSIM_ir':>8} | {'SSIM_vis':>8} | {'Qabf':>8}"
print(header)
print("-" * 95)

for name, suffix in zip(NAMES, EXPERIMENTS):
    result_dir = f"results_{suffix}"
    metrics_list = []

    # MSRS：ir 与 vi 同名对应，按序生成 1001.png / 1010.png ...
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
                pass

    if metrics_list:
        avg = {k: np.mean([r[k] for r in metrics_list]) for k in metrics_list[0].keys()}
        print(f"{name:>14} | {avg['EN']:>7.4f} | {avg['MI']:>7.4f} | {avg['SD']:>7.2f} | "
              f"{avg['SF']:>7.2f} | {avg['SSIM_ir']:>8.4f} | {avg['SSIM_vis']:>8.4f} | {avg['Qabf']:>8.4f}")
    else:
        print(f"{name:>14} | [NO RESULTS]")

print("=" * 95)
print("\n说明：EN/MI/SD/SF/SSIM/Qabf 越高越好，Qabf 正常范围 [0, 1]")
print("=" * 95 + "\n")
