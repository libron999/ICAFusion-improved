class args():
    epochs = 16
    batch_size = 4

    datasets_prepared = False

    # 训练数据路径（MSRS 数据集，ir 与 vi 文件名一一对应）
    # train 1083 对 / test 361 对，图像原始尺寸 640x480
    train_ir = './MSRS/train/ir'
    train_vi = './MSRS/train/vi'

    hight = 128
    width = 128
    image_size = 128

    save_model_dir = "models_training_5"
    save_loss_dir = "loss"

    cuda = 1

    g_lr = 0.0001
    d_lr = 0.0004
    log_interval = 5
    log_iter = 1
