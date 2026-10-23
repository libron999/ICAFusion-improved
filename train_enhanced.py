import random
from args import args
from Models_enhanced import Generator_Enhanced, D_IR, D_VI
import torch
import torch.optim as optim
from torch.autograd import Variable
from loss_enhanced import g_content_loss
import time
from tqdm import tqdm, trange
import numpy as np
import os
from utils_enhanced import make_floor
import utils_enhanced as utils

"""
增强版训练流程（支持 Generator_Enhanced 的多种配置）
使用方法：
  在 main_enhanced.py 中调用 train_enhanced.train(...) 并传入配置参数
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_grad(g_optimizer, dir_optimizer, dvi_optimizer):
    dir_optimizer.zero_grad()
    dvi_optimizer.zero_grad()
    g_optimizer.zero_grad()


def _finite(loss):
    """NaN/Inf 守卫：损失非有限值时跳过该步更新，防止一个坏 batch 毒化整个训练"""
    return not (torch.isnan(loss).any() or torch.isinf(loss).any())


def train(train_data_ir, train_data_vi,
          use_hfp=False, use_legm=False, use_mgdc=False, use_ffcm=False,
          save_suffix="enhanced", color=False):
    """
    训练函数，支持传入模块开关参数。
    save_suffix: 模型保存目录后缀，用于区分不同实验配置
    color: True=RGB三通道训练（红外图自动复制为三通道），False=原始灰度
    """
    models_save_path = make_floor(os.getcwd(), args.save_model_dir + "_" + save_suffix)
    print(models_save_path)
    loss_save_path = make_floor(models_save_path, args.save_loss_dir)
    print(loss_save_path)

    ch = 3 if color else 1

    # 初始化增强版生成器
    G = Generator_Enhanced(
        use_hfp=use_hfp,
        use_legm=use_legm,
        use_mgdc=use_mgdc,
        use_ffcm=use_ffcm,
        in_channels=ch
    ).to(device)

    D_ir = D_IR(in_channels=ch).to(device)
    D_vi = D_VI(in_channels=ch).to(device)
    g_content_criterion = g_content_loss().to(device)

    # 预热判别器：让延迟创建的 Linear 输出层先实例化并注册，
    # 确保下面的优化器能包含它的参数（修复原版 D 输出层从未被训练的 bug）
    with torch.no_grad():
        _dummy = torch.zeros(1, ch, args.hight, args.width, device=device)
        D_ir(_dummy)
        D_vi(_dummy)

    optimizerG = optim.Adam(G.parameters(), args.g_lr)
    optimizerD_ir = optim.Adam(D_ir.parameters(), args.d_lr)
    optimizerD_vi = optim.Adam(D_vi.parameters(), args.d_lr)

    print("\n===== Model Configuration =====")
    print(f"use_hfp  = {use_hfp}")
    print(f"use_legm = {use_legm}")
    print(f"use_mgdc = {use_mgdc}")
    print(f"use_ffcm = {use_ffcm}")
    print(f"color    = {color} (channels={ch})")
    print(f"Total G params: {sum(p.numel() for p in G.parameters())}")
    print("===============================\n")

    tbar = trange(args.epochs)

    ir_d_loss_lst = []
    vi_d_loss_lst = []
    g_adversarial_loss_lst = []
    content_loss_lst = []
    all_F_loss_lst = []
    all_L1_loss_lst = []
    g_loss_lst = []

    nan_skips_dvi = 0  # NaN 守卫跳过的更新步数统计（按更新类型分）

    all_ir_d_loss = 0
    all_vi_d_loss = 0
    all_g_adversarial_loss = 0.
    all_content_loss = 0.
    all_L1_loss = 0.
    all_F_loss = 0.

    for epoch in tbar:
        print('Epoch %d.....' % epoch)
        G.train()
        D_ir.train()
        D_vi.train()
        batch_size = args.batch_size
        image_set_ir, image_set_vi, batches = utils.load_dataset(
            train_data_ir, train_data_vi, batch_size, num_imgs=None)

        count = 0
        for batch in range(batches):
            count += 1
            reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
            img_model = 'RGB' if color else 'L'

            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]

            img_ir = utils.get_train_images_auto(image_paths_ir, height=args.hight, width=args.width, mode=img_model)
            img_vi = utils.get_train_images_auto(image_paths_vi, height=args.hight, width=args.width, mode=img_model)

            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)

            gamma_ = 10
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)

            img_fusion = G(img_ir, img_vi)

            # ---------- 更新 D_vi ----------
            for _ in range(2):
                D_out_vi = D_vi(img_vi)
                D_loss_vi = - torch.mean(D_out_vi)
                D_out_f = D_vi(img_fusion.detach())
                D_loss_f = D_out_f.mean()

                alpha_vi = torch.rand(img_vi.size(0), 1, 1, 1).to(device).expand_as(img_vi)
                interpolated_vi = Variable(alpha_vi * img_vi.data + (1 - alpha_vi) * img_fusion.data, requires_grad=True)
                Dvi_interpolated = D_vi(interpolated_vi)
                grad_vi = torch.autograd.grad(outputs=Dvi_interpolated,
                                              inputs=interpolated_vi,
                                              grad_outputs=torch.ones(Dvi_interpolated.size()).to(device),
                                              retain_graph=True,
                                              create_graph=True,
                                              only_inputs=True)[0]
                grad_vi = grad_vi.view(grad_vi.size(0), -1)
                grad_vi_l2norm = torch.sqrt(torch.sum(grad_vi ** 2, dim=1))
                Dvi_penalty = torch.mean((grad_vi_l2norm - 1) ** 2)

                vi_d_loss = D_loss_vi + D_loss_f + Dvi_penalty * gamma_
                if not _finite(vi_d_loss):
                    nan_skips_dvi += 1
                    reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                    continue
                all_vi_d_loss += vi_d_loss.item()

                reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                vi_d_loss.backward(retain_graph=True)
                gn = torch.nn.utils.clip_grad_norm_(D_vi.parameters(), max_norm=5.0)
                if not torch.isfinite(gn):
                    # 损失有限但梯度NaN（权重污染的唯一残留通道），跳过本步
                    nan_skips_dvi += 1
                    reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                    continue
                optimizerD_vi.step()

            # ---------- 更新 D_ir ----------
            for _ in range(2):
                D_out_ir = D_ir(img_ir)
                D_loss_ir = - torch.mean(D_out_ir)
                D_out_f = D_ir(img_fusion.detach())
                D_loss_f = D_out_f.mean()

                alpha_ir = torch.rand(img_ir.size(0), 1, 1, 1).to(device).expand_as(img_ir)
                interpolated_ir = Variable(alpha_ir * img_ir.data + (1 - alpha_ir) * img_fusion.data, requires_grad=True)
                Dir_interpolated = D_ir(interpolated_ir)
                grad_ir = torch.autograd.grad(outputs=Dir_interpolated,
                                              inputs=interpolated_ir,
                                              grad_outputs=torch.ones(Dir_interpolated.size()).to(device),
                                              retain_graph=True,
                                              create_graph=True,
                                              only_inputs=True)[0]
                grad_ir = grad_ir.view(grad_ir.size(0), -1)
                grad_ir_l2norm = torch.sqrt(torch.sum(grad_ir ** 2, dim=1))
                Dir_penalty = torch.mean((grad_ir_l2norm - 1) ** 2)

                ir_d_loss = D_loss_ir + D_loss_f + Dir_penalty * gamma_
                if not _finite(ir_d_loss):
                    nan_skips_dir += 1
                    reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                    continue
                all_ir_d_loss += ir_d_loss.item()

                reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                ir_d_loss.backward(retain_graph=True)
                gn = torch.nn.utils.clip_grad_norm_(D_ir.parameters(), max_norm=5.0)
                if not torch.isfinite(gn):
                    nan_skips_dir += 1
                    reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                    continue
                optimizerD_ir.step()

            # ---------- 更新 G ----------
            img_fusion = G(img_ir, img_vi)
            content_loss, L1_loss, F_loss = g_content_criterion(img_ir, img_vi, img_fusion)

            dir_g_adversarial_loss = -D_ir(img_fusion).mean()
            dvi_g_adversarial_loss = -D_vi(img_fusion).mean()
            g_adversarial_loss = (dir_g_adversarial_loss + dvi_g_adversarial_loss)
            lambda_ = 1
            g_loss = lambda_ * g_adversarial_loss + content_loss

            if not _finite(g_loss):
                nan_skips_g += 1
                reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                continue
            all_F_loss += F_loss.item()
            all_L1_loss += L1_loss.item()
            all_g_adversarial_loss += g_adversarial_loss.item()
            all_content_loss += content_loss.item()

            reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
            g_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
            if not torch.isfinite(gn):
                nan_skips_g += 1
                reset_grad(optimizerG, optimizerD_ir, optimizerD_vi)
                continue
            optimizerG.step()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tepoch {}:[{}/{}]\n " \
                       "ir_d_loss: {:.6}\t vi_d_loss: {:.6}" \
                       "\t g_adversarial_loss:{:.6}\t content_loss:{:.6}\t g_loss:{:.6}" \
                       "\t L1_loss:{:.6}\t  F_loss:{:.6}".format(
                    time.ctime(), epoch + 1, count, batches,
                    all_ir_d_loss / args.log_interval, all_vi_d_loss / args.log_interval,
                    all_g_adversarial_loss / args.log_interval, all_content_loss / args.log_interval,
                    (all_g_adversarial_loss + all_content_loss) / args.log_interval,
                    all_L1_loss / args.log_interval, all_F_loss / args.log_interval
                )
                tbar.set_description(mesg)

                ir_d_loss_lst.append(all_ir_d_loss / args.log_interval)
                vi_d_loss_lst.append(all_vi_d_loss / args.log_interval)
                g_adversarial_loss_lst.append(all_g_adversarial_loss / args.log_interval)
                content_loss_lst.append(all_content_loss / args.log_interval)
                all_L1_loss_lst.append(all_L1_loss / args.log_interval)
                all_F_loss_lst.append(all_F_loss / args.log_interval)
                g_loss_lst.append((all_g_adversarial_loss + all_content_loss) / args.log_interval)

                all_ir_d_loss = 0
                all_vi_d_loss = 0
                all_g_adversarial_loss = 0.
                all_L1_loss = 0
                all_content_loss = 0.
                all_F_loss = 0

        total_skips = nan_skips_dvi + nan_skips_dir + nan_skips_g
        if total_skips > 0:
            print(f"[WARN] epoch {epoch}: 跳过 {total_skips} 个 NaN/Inf 更新步 "
                  f"(D_vi:{nan_skips_dvi} D_ir:{nan_skips_dir} G:{nan_skips_g})")
            nan_skips_dvi = nan_skips_dir = nan_skips_g = 0

        # ----- 保存模型 -----
        if (epoch + 1) % args.log_iter == 0:
            G.eval()
            G.to(device)
            G_save_model_filename = "G_Epoch_" + str(epoch) + ".model"
            G_model_path = os.path.join(models_save_path, G_save_model_filename)
            torch.save(G.state_dict(), G_model_path)

            for name, data in [
                ("ir_d_loss", ir_d_loss_lst),
                ("vi_d_loss", vi_d_loss_lst),
                ("g_adversarial_loss", g_adversarial_loss_lst),
                ("content_loss", content_loss_lst),
                ("all_L1_loss", all_L1_loss_lst),
                ("all_F_loss", all_F_loss_lst),
                ("g_loss", g_loss_lst),
            ]:
                part = np.array(data)
                loss_filename_path = f"{name}_epoch_{epoch}.npy"
                save_loss_path = os.path.join(loss_save_path, loss_filename_path)
                np.save(save_loss_path, part)

    # 保存最终模型
    G.eval()
    G.to(device)
    G_save_model_filename = "Final_G_Epoch_" + str(epoch) + ".model"
    G_model_path = os.path.join(models_save_path, G_save_model_filename)
    torch.save(G.state_dict(), G_model_path)
    print("\nDone, trained Final_G_model saved at", G_model_path)
