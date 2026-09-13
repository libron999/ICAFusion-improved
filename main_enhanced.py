import time
from args import args
import utils_enhanced as utils
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

"""
增强版入口脚本
支持切换：原始模型 / 增强版模型
支持配置：HFP / LEGM / MGDC / FFCM 模块开关
"""

# ==================== 配置区域 ====================
# 模式选择：0=测试, 1=训练
flag = 0

# 是否使用增强版模型
USE_ENHANCED = True

# 增强模块开关（仅在 USE_ENHANCED=True 时有效）
# 全部关闭 = 原始 ICAFusion Baseline
USE_HFP = False     # 高频感知模块
USE_LEGM = False    # 频感网络单元
USE_MGDC = False    # 多尺度分组空洞卷积
USE_FFCM = False    # 融合傅里叶卷积混合器

# 模型保存后缀（用于区分不同实验）
SAVE_SUFFIX = "baseline"

# 测试时加载的模型路径
TEST_MODEL_PATH = "./models_training_5_baseline/Final_G_Epoch_49.model"
# 测试结果保存目录
RESULT_DIR = "results_baseline"

# 测试数据集选择："MSRS"（361对，同名配对）或 "TNO"（25对，IR{i}/VIS{i}命名）
TEST_DATASET = "TNO"
# =================================================


if flag == 1:
    IS_TRAINING = True
else:
    IS_TRAINING = False


def main():
    if IS_TRAINING:
        # -------------------- 训练模式 --------------------
        data_dir_ir = utils.list_images(args.train_ir)
        data_dir_vi = utils.list_images(args.train_vi)
        train_data_ir = data_dir_ir
        train_data_vi = data_dir_vi

        print("\ntrain_data_ir num is ", len(train_data_ir))
        print("train_data_vi num is ", len(train_data_vi))

        if USE_ENHANCED:
            from Models_enhanced import Generator_Enhanced
            import train_enhanced

            print("\n===== Using Enhanced Generator =====")
            print(f"HFP  : {USE_HFP}")
            print(f"LEGM : {USE_LEGM}")
            print(f"MGDC : {USE_MGDC}")
            print(f"FFCM : {USE_FFCM}")
            print(f"Save suffix: {SAVE_SUFFIX}")
            print("====================================\n")

            train_enhanced.train(
                train_data_ir, train_data_vi,
                use_hfp=USE_HFP,
                use_legm=USE_LEGM,
                use_mgdc=USE_MGDC,
                use_ffcm=USE_FFCM,
                save_suffix=SAVE_SUFFIX
            )
        else:
            from Models import Generator
            import train
            print("\n===== Using Original Generator (Baseline) =====\n")
            train.train(train_data_ir, train_data_vi)

    else:
        # -------------------- 测试模式 --------------------
        print("\nBegin to generate pictures ...\n")

        # 按配置构建测试图像对列表
        if TEST_DATASET == "MSRS":
            test_ir_dir = "./MSRS/test/ir/"
            test_vi_dir = "./MSRS/test/vi/"
            ir_files = sorted(f for f in os.listdir(test_ir_dir) if f.endswith('.png'))
            pairs = [(test_ir_dir + f, test_vi_dir + f) for f in ir_files]
        else:  # TNO：IR{i}.png / VIS{i}.png，跳过不存在的编号
            test_dir = "./test_imgs/tno/"
            pairs = []
            for i in range(1, 100):
                ir_p = test_dir + f"IR{i}.png"
                if os.path.exists(ir_p):
                    pairs.append((ir_p, test_dir + f"VIS{i}.png"))
        print(f'{TEST_DATASET} dataset begin to test, {len(pairs)} pairs')

        if USE_ENHANCED:
            from Models_enhanced import Generator_Enhanced
            model = Generator_Enhanced(
                use_hfp=USE_HFP,
                use_legm=USE_LEGM,
                use_mgdc=USE_MGDC,
                use_ffcm=USE_FFCM
            )
        else:
            from Models import Generator
            model = Generator()

        model.load_state_dict(torch.load(TEST_MODEL_PATH), strict=False)
        print('# generator parameters:', sum(param.numel() for param in model.parameters()))
        model.eval()
        model.to(device)

        with torch.no_grad():
            from generate_enhanced import generate
            begin = time.time()
            for index, (ir_path, vis_path) in enumerate(pairs, start=1):
                generate(model, ir_path, vis_path, RESULT_DIR, index, mode='L')
            end = time.time()
            print("consumption time of generating:%s " % (end - begin))


if __name__ == "__main__":
    main()
