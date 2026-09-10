#!/usr/bin/env python3
"""
测试脚本：验证 Models_enhanced.py 中所有模块组合可正常运行
运行方式：python test_models.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from Models_enhanced import Generator_Enhanced, HFP, LEGM, MGDC, FFCM


def test_module(module_class, name, *args, **kwargs):
    """测试单个模块的前向传播"""
    try:
        m = module_class(*args, **kwargs).cpu()
        x = torch.randn(1, args[0] if args else kwargs.get('in_channels', 32), 64, 64)
        with torch.no_grad():
            out = m(x)
        assert out.shape[0] == 1
        print(f"  [OK] {name}: input {x.shape} -> output {out.shape}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return False


def test_generator(config_name, **kwargs):
    """测试生成器的前向传播和反向传播"""
    try:
        G = Generator_Enhanced(**kwargs).cpu()
        ir = torch.randn(1, 1, 128, 128)
        vi = torch.randn(1, 1, 128, 128)

        # 前向传播
        with torch.no_grad():
            out = G(ir, vi)
        assert out.shape == (1, 1, 128, 128), f"输出形状异常: {out.shape}"

        # 反向传播测试
        out2 = G(ir, vi)
        loss = out2.mean()
        loss.backward()

        params = sum(p.numel() for p in G.parameters())
        print(f"  [OK] Generator ({config_name}): params={params:,}, forward/backward OK")
        return True
    except Exception as e:
        print(f"  [FAIL] Generator ({config_name}): {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  Models_enhanced.py 运行时测试")
    print("=" * 60)

    all_ok = True

    # 1. 测试单个模块
    print("\n[1] 独立模块测试")
    all_ok &= test_module(HFP, "HFP(64)", 64)
    all_ok &= test_module(HFP, "HFP(32)", 32)
    all_ok &= test_module(HFP, "HFP(16)", 16)
    all_ok &= test_module(LEGM, "LEGM(384)", 384)
    all_ok &= test_module(LEGM, "LEGM(192)", 192)
    all_ok &= test_module(MGDC, "MGDC(16->16)", 16, 16)
    all_ok &= test_module(MGDC, "MGDC(16->32)", 16, 32)
    all_ok &= test_module(FFCM, "FFCM(128)", 128)

    # 2. 测试生成器各种配置
    print("\n[2] 生成器组合测试")
    configs = [
        ("Baseline", dict(use_hfp=False, use_legm=False, use_mgdc=False, use_ffcm=False)),
        ("HFP only", dict(use_hfp=True, use_legm=False, use_mgdc=False, use_ffcm=False)),
        ("LEGM only", dict(use_hfp=False, use_legm=True, use_mgdc=False, use_ffcm=False)),
        ("MGDC only", dict(use_hfp=False, use_legm=False, use_mgdc=True, use_ffcm=False)),
        ("FFCM only", dict(use_hfp=False, use_legm=False, use_mgdc=False, use_ffcm=True)),
        ("HFP+LEGM", dict(use_hfp=True, use_legm=True, use_mgdc=False, use_ffcm=False)),
        ("HFP+MGDC", dict(use_hfp=True, use_legm=False, use_mgdc=True, use_ffcm=False)),
        ("MGDC+FFCM", dict(use_hfp=False, use_legm=False, use_mgdc=True, use_ffcm=True)),
        ("All modules", dict(use_hfp=True, use_legm=True, use_mgdc=True, use_ffcm=True)),
    ]

    for name, cfg in configs:
        all_ok &= test_generator(name, **cfg)

    # 3. 总结
    print("\n" + "=" * 60)
    if all_ok:
        print("  全部测试通过！代码可以正常运行。")
    else:
        print("  部分测试失败，请查看上方错误信息。")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
