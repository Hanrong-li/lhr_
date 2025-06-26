import os
import time
import math
import logging
import jittor as jt
from jittor import nn
from jittor import optim
from jittor.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import psutil

# 导入转换后的模型和数据集
from model import Transformer, ModelArgs
from data_set import PretrainDataset

# 初始化 Jittor
jt.flags.use_cuda = 1  # 使用GPU
jt.flags.log_silent = True  # 减少日志输出


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch):
    start_time = time.time()
    model.train()
    epoch_losses = []  # 存储当前epoch的loss值
    for step, (X, Y) in enumerate(train_loader):
        # 更新学习率
        current_iter = epoch * iter_per_epoch + step
        lr = get_lr(current_iter) if decay_lr else learning_rate
        optimizer.lr = lr

        # 前向传播和损失计算
        with jt.flag_scope(amp_level=amp_level):
            logits = model(X)
            loss = nn.cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
                ignore_index=-1
            )
            loss = loss / gradient_accumulation_steps

        # 反向传播
        optimizer.backward(loss)

        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if grad_clip != 0.0:
                optimizer.clip_grad_norm(grad_clip, norm_type=2)

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 记录损失值
            loss_value = loss.item() * gradient_accumulation_steps
            epoch_losses.append(loss_value)
            loss_records.append((current_iter, loss_value))

        # 打印日志
        if step % log_interval == 0:
            spend_time = time.time() - start_time
            avg_time = spend_time / (step + 1)
            eta = avg_time * (iter_per_epoch - step) / 60
            logger.info(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} ETA:{:.1f}min'.format(
                    epoch, max_epoch, step, iter_per_epoch,
                    loss.item() * gradient_accumulation_steps, lr, eta))

        # 保存检查点
        if step % save_interval == 0:
            model_path = os.path.join(save_dir, f'iter_{current_iter}.pkl')
            model.save(model_path)
            logger.info(f'Saved model to {model_path}')

def init_model():
    # 模型配置
    model_args = ModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    # 初始化模型
    if init_from == "scratch":
        logger.info("Initializing a new model from scratch")
        model = Transformer(model_args)
    elif init_from == "resume":
        logger.info(f"Resuming training from {out_dir}")
        model_path = os.path.join(out_dir, "ckpt.pkl")
        model = Transformer.load(model_path)

    return model

def plot_loss_curve(loss_records, save_dir):
    if not loss_records:
        logger.warning("No loss records to plot.")
        return
    
    steps, losses = zip(*loss_records)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制原始loss点
    plt.scatter(steps, losses, s=5, alpha=0.5, label='Step Loss')
    
    # 计算并绘制滑动平均loss
    window_size = max(1, len(losses) // 20)  # 自适应窗口大小
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    plt.plot(steps[window_size-1:], smoothed_losses, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
    
    plt.title('Training Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plot_path = os.path.join(save_dir, 'loss_curve_bigger.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss curve saved to {plot_path}")

if __name__ == "__main__":
    # ================ 配置参数 ================
    out_dir = 'out'
    max_epoch = 1
    log_interval = 100
    save_interval = 10000
    init_from = 'scratch'
    gradient_accumulation_steps = 4
    batch_size = 16
    max_seq_len = 256
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    dropout = 0.0
    vocab_size = 64793  # 词汇表大小
    learning_rate = 3e-4
    weight_decay = 1e-1
    beta1= 0.9
    beta2= 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = 80000
    min_lr = 1e-5
    amp_level = 2  # Jittor混合精度级别 (0=float32, 3=bfloat16)

    # 创建输出目录
    save_dir = os.path.join(out_dir, 'pretrain')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化日志
    logger = get_logger(os.path.join(save_dir, 'log.log'))

    # 记录配置参数
    config_params = {
        'out_dir': out_dir,
        'max_epoch': max_epoch,
        'log_interval': log_interval,
        'save_interval': save_interval,
        'init_from': init_from,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'dim': dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'multiple_of': multiple_of,
        'dropout': dropout,
        'vocab_size': vocab_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'grad_clip': grad_clip,
        'decay_lr': decay_lr,
        'warmup_iters': warmup_iters,
        'lr_decay_iters': lr_decay_iters,
        'min_lr': min_lr,
        'amp_level': amp_level,
    }

    logger.info("Training Configuration:")
    for key, value in config_params.items():
        logger.info(f"  {key}: {value}")

    # 设置混合精度
    if amp_level > 0:
        jt.flags.amp_level = amp_level

    # 初始化模型
    model = init_model()

    # 初始化优化器
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))

    # 初始化数据集
    data_path_list = [
        './data/baidubaike_563w_1.bin',
        './data/baidubaike_563w_2.bin',
         './data/wiki.bin'
    ]
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)

    # 初始化数据加载器
    train_loader = train_ds.set_attrs(
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    # 计算迭代次数
    iter_per_epoch = len(train_loader)
    logger.info(f"Training dataset size: {len(train_ds)}")
    logger.info(f"Iterations per epoch: {iter_per_epoch}")
    logger.info(f"Total iterations: {iter_per_epoch * max_epoch}")
    
    loss_records = []  # 存储(step, loss)元组
    # 训练循环
    for epoch in range(max_epoch):
        logger.info(f"Starting epoch {epoch + 1}/{max_epoch}")
        train_epoch(epoch)

        # 保存最终模型
        model_path = os.path.join(save_dir, f'epoch_bigger{epoch}.pkl')
        model.save(model_path)
        logger.info(f'Saved final model to {model_path}')

    logger.info("Training completed!")
    #绘制loss曲线
    plot_loss_curve(loss_records, save_dir)