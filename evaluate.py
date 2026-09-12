"""
图像融合质量评估脚本
计算指标：EN, MI, SD, SF, SSIM, Qabf

用法：
    python evaluate.py --ir_dir ./MSRS/test/ir/ --vis_dir ./MSRS/test/vi/ \
                       --fus_dir ./results_baseline/ --num 361
"""

import os
import argparse
import numpy as np
from PIL import Image


def read_gray(path, size=(128, 128)):
    """读取灰度图像为 numpy 数组，resize 到指定尺寸，值域 [0, 255]"""
    img = Image.open(path).convert('L')
    if size is not None:
        img = img.resize(size, Image.NEAREST)
    return np.array(img, dtype=np.float64)
    """读取灰度图像为 numpy 数组，值域 [0, 255]"""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float64)


def entropy(img):
    """信息熵 EN"""
    hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def mutual_information(img_a, img_b, img_f):
    """互信息 MI = MI(A,F) + MI(B,F)"""
    def _mi(x, y):
        hist_2d, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=256, range=[[0, 256], [0, 256]])
        hist_2d = hist_2d / (np.sum(hist_2d) + 1e-10)
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)
        px_py = np.outer(px, py)
        mask = hist_2d > 0
        return np.sum(hist_2d[mask] * np.log2(hist_2d[mask] / (px_py[mask] + 1e-10) + 1e-10))
    return _mi(img_a, img_f) + _mi(img_b, img_f)


def std_dev(img):
    """标准差 SD"""
    return np.std(img)


def spatial_frequency(img):
    """空间频率 SF"""
    rf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    cf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    return np.sqrt(rf ** 2 + cf ** 2)


def ssim(img_a, img_b, K1=0.01, K2=0.03, L=255):
    """简化版 SSIM（全局平均）"""
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_a = np.mean(img_a)
    mu_b = np.mean(img_b)
    sigma_a = np.std(img_a)
    sigma_b = np.std(img_b)
    sigma_ab = np.mean((img_a - mu_a) * (img_b - mu_b))
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2)
    return numerator / (denominator + 1e-10)


def qabf(img_a, img_b, img_f):
    """
    Qabf - 基于梯度的融合质量指标 (Petrovic & Xydeas, 2000)
    纯 numpy 实现
    """
    def _conv2d(img, kernel):
        """简单的 2D 卷积（same padding）"""
        h, w = img.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return result

    def _sobel(img):
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        g_x = _conv2d(img, gx)
        g_y = _conv2d(img, gy)
        g = np.sqrt(g_x ** 2 + g_y ** 2)
        a = np.arctan2(g_y, g_x)
        return g, a

    gA, aA = _sobel(img_a)
    gB, aB = _sobel(img_b)
    gF, aF = _sobel(img_f)

    # 梯度强度保留率 (0~1)，标准公式
    gAF = np.where((gA + gF) > 0, 2 * gA * gF / (gA**2 + gF**2 + 1e-10), 0)
    gBF = np.where((gB + gF) > 0, 2 * gB * gF / (gB**2 + gF**2 + 1e-10), 0)

    # 梯度方向保留率 (0~1)
    # 梯度方向差取模 π（相反方向梯度对应同一边缘）
    diffAF = np.abs(aA - aF)
    diffAF = np.minimum(diffAF, np.pi - diffAF)
    dAF = 1 - diffAF / (np.pi / 2)
    dAF = np.clip(dAF, 0, 1)

    diffBF = np.abs(aB - aF)
    diffBF = np.minimum(diffBF, np.pi - diffBF)
    dBF = 1 - diffBF / (np.pi / 2)
    dBF = np.clip(dBF, 0, 1)

    # 质量度量
    qAF = gAF * dAF
    qBF = gBF * dBF

    # 显著性权重
    wA = gA ** 2
    wB = gB ** 2
    w = wA + wB + 1e-10

    q = (wA * qAF + wB * qBF) / w
    return np.mean(q)
    """
    Qabf - 基于梯度的融合质量指标 (Petrovic & Xydeas, 2000)
    纯 numpy 实现，无外部依赖
    """
    def _conv2d(img, kernel):
        """简单的 2D 卷积（same padding）"""
        h, w = img.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        result = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return result

    def _sobel(img):
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        g_x = _conv2d(img, gx)
        g_y = _conv2d(img, gy)
        g = np.sqrt(g_x ** 2 + g_y ** 2)
        a = np.arctan2(g_y, g_x)
        return g, a

    gA, aA = _sobel(img_a)
    gB, aB = _sobel(img_b)
    gF, aF = _sobel(img_f)

    # 相对梯度强度
    gAF = np.where(gA > gF, gF / (gA + 1e-10), gA / (gF + 1e-10))
    gBF = np.where(gB > gF, gF / (gB + 1e-10), gB / (gF + 1e-10))

    # 相对梯度方向差
    dAF = 1 - np.abs(aA - aF) / (np.pi / 2 + 1e-10)
    dBF = 1 - np.abs(aB - aF) / (np.pi / 2 + 1e-10)

    # 质量度量
    qAF = gAF * dAF
    qBF = gBF * dBF

    # 显著性权重
    wA = gA ** 2
    wB = gB ** 2
    w = wA + wB + 1e-10

    q = (wA * qAF + wB * qBF) / w
    return np.mean(q)


def evaluate_pair(ir_path, vis_path, fus_path):
    """评估单对图像"""
    ir = read_gray(ir_path)
    vis = read_gray(vis_path)
    fus = read_gray(fus_path)

    return {
        'EN': entropy(fus),
        'MI': mutual_information(ir, vis, fus),
        'SD': std_dev(fus),
        'SF': spatial_frequency(fus),
        'SSIM_ir': ssim(ir, fus),
        'SSIM_vis': ssim(vis, fus),
        'Qabf': qabf(ir, vis, fus),
    }


def main():
    parser = argparse.ArgumentParser(description='Image Fusion Quality Evaluation')
    parser.add_argument('--ir_dir', default='./MSRS/test/ir/', help='红外图像目录')
    parser.add_argument('--vis_dir', default='./MSRS/test/vi/', help='可见光图像目录')
    parser.add_argument('--fus_dir', default='./results_baseline/', help='融合结果目录')
    parser.add_argument('--num', type=int, default=361, help='测试图像对数（<=0 表示全部）')
    args = parser.parse_args()

    # MSRS：ir 与 vi 文件名一一对应（同名），按文件名排序后按序配对
    ir_files = sorted(f for f in os.listdir(args.ir_dir) if f.endswith('.png'))
    if args.num > 0:
        ir_files = ir_files[:args.num]

    results = []
    header = f"{'Idx':>4} | {'EN':>8} | {'MI':>8} | {'SD':>10} | {'SF':>8} | {'SSIM_ir':>8} | {'SSIM_vis':>8} | {'Qabf':>8}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print(header)
    print(sep)

    for i, ir_name in enumerate(ir_files, start=1):
        ir_path = os.path.join(args.ir_dir, ir_name)
        vis_path = os.path.join(args.vis_dir, ir_name)
        fus_name = f"100{i}.png" if i < 10 else f"10{i}.png"
        fus_path = os.path.join(args.fus_dir, fus_name)

        if not os.path.exists(fus_path):
            print(f"  [SKIP] Fusion result not found: {fus_path}")
            continue

        metrics = evaluate_pair(ir_path, vis_path, fus_path)
        results.append(metrics)

        print(f"{i:>4} | {metrics['EN']:>8.4f} | {metrics['MI']:>8.4f} | "
              f"{metrics['SD']:>10.2f} | {metrics['SF']:>8.2f} | "
              f"{metrics['SSIM_ir']:>8.4f} | {metrics['SSIM_vis']:>8.4f} | "
              f"{metrics['Qabf']:>8.4f}")

    if not results:
        print("[ERROR] No valid fusion results found. Please run testing first.")
        return

    avg = {k: np.mean([r[k] for r in results]) for k in results[0].keys()}

    print(sep)
    print(f"{'AVG':>4} | {avg['EN']:>8.4f} | {avg['MI']:>8.4f} | "
          f"{avg['SD']:>10.2f} | {avg['SF']:>8.2f} | "
          f"{avg['SSIM_ir']:>8.4f} | {avg['SSIM_vis']:>8.4f} | "
          f"{avg['Qabf']:>8.4f}")
    print("=" * len(header))

    print("\n========== 平均指标汇总 ==========")
    print(f"  EN (信息熵)      : {avg['EN']:.4f}")
    print(f"  MI (互信息)      : {avg['MI']:.4f}")
    print(f"  SD (标准差)      : {avg['SD']:.2f}")
    print(f"  SF (空间频率)    : {avg['SF']:.2f}")
    print(f"  SSIM_ir          : {avg['SSIM_ir']:.4f}")
    print(f"  SSIM_vis         : {avg['SSIM_vis']:.4f}")
    print(f"  Qabf             : {avg['Qabf']:.4f}")
    print("==================================\n")


if __name__ == '__main__':
    main()
