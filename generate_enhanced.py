from args import args
import utils_enhanced as utils
from torch.autograd import Variable
from Models_enhanced import Generator_Enhanced
from utils_enhanced import make_floor
from PIL import Image
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

"""
增强版测试/推理流程（兼容新版 scipy）
使用 Pillow 替代 scipy.misc.imsave
"""


def _generate_fusion_image(G_model, ir_img, vis_img):
    """封装生成器的前向传播：输入红外和可见光图像，输出融合图像。"""
    f = G_model(ir_img, vis_img)
    return f


def load_model(model_path, use_hfp=False, use_legm=False, use_mgdc=False, use_ffcm=False):
    """加载训练好的增强版生成器权重"""
    G_model = Generator_Enhanced(
        use_hfp=use_hfp,
        use_legm=use_legm,
        use_mgdc=use_mgdc,
        use_ffcm=use_ffcm
    )
    G_model.load_state_dict(torch.load(model_path), strict=False)
    print('# generator parameters:', sum(p.numel() for p in G_model.parameters()))
    G_model.eval()
    G_model.to(device)
    return G_model


def generate(model, ir_path, vis_path, result, index, mode):
    """
    对单对图像进行融合，并保存结果。
    测试时使用与训练相同的尺寸 (128x128) 以确保兼容性。
    """
    # 使用训练时相同的尺寸读取图像
    ir_img = utils.get_test_images(ir_path, height=args.hight, width=args.width, mode=mode)
    vis_img = utils.get_test_images(vis_path, height=args.hight, width=args.width, mode=mode)
    ir_img = ir_img.to(device)
    vis_img = vis_img.to(device)
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)

    img_fusion = _generate_fusion_image(model, ir_img, vis_img)
    img_fusion = (img_fusion / 2 + 0.5) * 255

    # 取 batch 第 0 张，去掉单通道维度，得到 [H, W]
    img = img_fusion[0].squeeze().cpu().clamp(0, 255).numpy()

    result_path = make_floor(os.getcwd(), result)

    if index < 10:
        f_filenames = "100" + str(index) + '.png'
    else:
        f_filenames = "10" + str(index) + '.png'

    output_path = result_path + '/' + f_filenames
    Image.fromarray(img.astype(np.uint8)).save(output_path)
