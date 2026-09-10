class args():
    epochs = 16
    batch_size = 4

    datasets_prepared = False

    # 训练数据路径（红外和可见光图像文件夹）
    # 当前使用 test_imgs 下的测试数据作为临时训练数据验证代码
    # 正式训练时请替换为你的真实训练数据集路径
    train_ir = r'C:\Users\leoli\PycharmProjects\ICAFusion\test_imgs\ir'
    train_vi = r'C:\Users\leoli\PycharmProjects\ICAFusion\test_imgs\vi'

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
