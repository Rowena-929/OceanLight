"""
Training script for GenCast.
"""
import os
import time

# from torch.utils.checkpoint import checkpoint

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch, gc  # noqa: E402
import lightning
import pytorch_lightning as L  # noqa: E402
import dgl
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import WandbLogger  # noqa: E402
from scipy.constants import precision
# from tensorboard.compat.tensorflow_stub.dtypes import float32
from torch.utils.data import DataLoader  # noqa: E402
import xarray as xr
import dgl
from pytorch_lightning.loggers import TensorBoardLogger  # 替换 WandbLogger
import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict
from weighted_mseloss import WeightedMSELoss

# from graph_weather.data.gencast_dataloader import GenCastDataset  # noqa: E402
from dataloader.predict_dataloader4 import TrainDataset, ValDataset
# from dataloader.unet_dataloader import TrainDataset, ValDataset
from model import FunctionalGenerativeNetwork as FGN
from unet import UNet
from ViT import VisionTransformer

# torch.set_float32_matmul_precision("high")

############################################## SETTINGS ############################################

# training settings
NUM_EPOCHS = 20
NUM_DEVICES = 2
NUM_ACC_GRAD = 1
INITIAL_LR = 1e-3
BATCH_SIZE = 1  # true batch size: BATCH_SIZE*NUM_DEVICES*NUM_ACC_GRAD
LEAD = 5
WARMUP = 1000
wand = True

# dataloader setting
NUM_WORKERS = 2  # 测试双进程DataLoader + dataset内部4线程并行
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

# model configs
CHECKPOINT_PATH = f"./checkpoints/mercator0.25total/lead{LEAD}/last/last-epoch=04.ckpt"
CFG = {
    "hidden_dims": [512, 512],
    "num_blocks": 16,
    "num_heads": 4,
    "splits": 6,
    "num_hops": 0,
    "sparse": False,
    "use_edges_features": True,
    "scale_factor": 1.0,
}

# dataset configs
surface_features = [
    "zos",
    "u",
    "v",
    "sst",
]
sea_features = [
    "thetao",
    "s",
    "u0",
    "v0",
]
static_features = [
    "depth",
    "land_sea_mask",
]

OBS_PATH = "dataset.zarr"
mask = torch.from_numpy(np.delete(np.load('./data/total0.25mask.npy'),[1,2,3],axis=1))

#################################################################################################

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine Scheduler with Warmup"""

    def __init__(self, optimizer, warmup, max_iters):
        """Initialize the scheduler"""
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        """Return the learning rates"""
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """Return the scaling factor for the learning rate at a given iteration"""
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class LitModel(L.LightningModule):
    """Lightning wrapper for Gencast"""

    def __init__(
        self,
        warmup,
        learning_rate,
        cosine_t_max,
        # pressure_levels,
        grid_lon,
        grid_lat,
        fine_lon,
        fine_lat,
        input_features_dim, #96
        output_features_dim, #96
        hidden_dims,
        num_blocks,
        num_heads,
        splits,
        num_hops,
        sparse,
        use_edges_features,
        scale_factor=1.0,
    ):
        """Initialize the module"""
        super().__init__()

        self.model = FGN(
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            fine_lon=fine_lon,
            fine_lat=fine_lat,
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            noise_dimension=32,
            hidden_dims=hidden_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            splits=splits,
            num_hops=num_hops,
            device=self.device,
            sparse=sparse,
            use_edges_features=use_edges_features,
            scale_factor=scale_factor,
        )

        self.criterion = WeightedMSELoss(
            grid_lat=torch.tensor(fine_lat).to(self.device),
            # pressure_levels=torch.tensor(pressure_levels).to(self.device),
            # num_atmospheric_features=len(atmospheric_features),
            # single_features_weights=torch.tensor([1.0, 0.1, 0.1, 0.1, 0.1]).to(self.device),
        )

        self.learning_rate = learning_rate
        self.cosine_t_max = cosine_t_max
        self.warmup = warmup
        self.LOSS_SPIKE_THRESHOLD = 10.0 # 可根据实际情况调整

    def forward(self, prev_inputs, num_ensemble):
        """Compute forward pass"""
        return self.model(prev_inputs, num_ensemble)

    def training_step(self, batch, batch_idx):
        """Single training step"""
        prev_inputs, target_residuals, indices = batch

        preds = self.model(
            previous_weather_state=prev_inputs,
            num_ensemble=1,
        )
        # preds = checkpoint(self.model, prev_inputs, 1, use_reentrant=False)
        loss = self.criterion(preds, target_residuals, torch.permute(mask, (0, 3, 2, 1)).to(self.device))
        # loss = self.criterion(preds, target_residuals, mask.to(self.device)).to(self.device)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        if loss.item() > self.LOSS_SPIKE_THRESHOLD:
            print(f"Loss spike detected at batch_idx: {batch_idx}, data indices: {indices}")
        # for i in range(torch.cuda.device_count()):
        #     peak_gb = torch.cuda.max_memory_reserved(i) / 1024 ** 3
        #     print(f"[Step {batch_idx}] GPU {i} 当前峰值显存: {peak_gb:.2f} GB", flush=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        prev_inputs, target_residuals, indices = batch

        preds = self.model(
            previous_weather_state=prev_inputs,
            num_ensemble=1,
        )
        val_loss = self.criterion(preds, target_residuals, torch.permute(mask, (0, 3, 2, 1)).to(self.device))
        # val_loss = self.criterion(preds, target_residuals, mask.to(self.device))

        if val_loss.item() > self.LOSS_SPIKE_THRESHOLD:
            print(f"Validation loss spike detected at batch_idx: {batch_idx}, data indices: {indices}")

        # 记录验证损失
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        """Initialize the optimizer"""
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.1, betas=(0.9, 0.95)
        )
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cosine_t_max)
        sch = CosineWarmupScheduler(opt, warmup=self.warmup, max_iters=self.cosine_t_max)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",  # step means "batch" here, default: epoch
                "frequency": 1,  # default
            },
        }

    def plot_sample(self, prev_inputs, target_residuals):
        """Plot 2m_temperature and geopotential"""
        global mask
        prev_inputs = prev_inputs[:1, :, :, :]
        target = target_residuals[:1, :, :, :]
        preds = self.model(prev_inputs, 1).detach()[0]
        mask = mask.to(self.device)
        # loss = (torch.abs(target - preds)).mean()
        # print(target[0, :, :, 5]*2.03+34.09, preds[0, :, :, 5]*2.03+34.09)
        sstloss = (torch.abs((target - preds)[0, :, :, 1]*11.69)*mask[0, 1].T).sum() / (mask[0, 1].sum())
        s0loss = (torch.abs((target - preds)[0, :, :, 2]*2.03)*mask[0, 2].T).sum() / (mask[0, 2].sum())

        fig1, ax = plt.subplots(2)
        ax[0].imshow((((preds[0, :, :, 1]+prev_inputs[0, :, :, 1])*11.69+13.76).T*mask[0, 1, :, :]).cpu(), origin="lower", cmap="coolwarm")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Diffusion sampling prediction")

        ax[1].imshow(((target[0, :, :, 1]+prev_inputs[0, :, :, 1])*11.69+13.76).T.cpu(), origin="lower", cmap="coolwarm")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Ground truth")
        ax[1].text(10, 10, f's0loss: {s0loss:.4f}',  # 位置在(10,10)处
                   color='white',  # 白色文字
                   bbox=dict(facecolor='black', alpha=0.5),  # 半透明黑色背景
                   fontsize=12)  # 字体大小
        ax[1].text(10, 30, f'SST Loss: {sstloss:.4f}',  # 位置在(10,30)处，y坐标增加了
                   color='white',
                   bbox=dict(facecolor='black', alpha=0.5),
                   fontsize=12)

        fig2, ax = plt.subplots(2)
        ax[0].imshow((((preds[0, :, :, 2]+prev_inputs[0, :, :, 2])*2.03+34.09).T*mask[0, 2, :, :]).cpu(), origin="lower", cmap="viridis")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Diffusion sampling prediction")

        ax[1].imshow(((target[0, :, :, 2]+prev_inputs[0, :, :, 2])*2.03+34.09).T.cpu(), origin="lower", cmap="viridis")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Ground truth")

        return fig1, fig2


class SamplingCallback(Callback):
    def __init__(self, data):
        prev_inputs, target_residuals, _ = data
        self.prev_inputs = torch.tensor(prev_inputs).unsqueeze(0)
        self.target_residuals = torch.tensor(target_residuals).unsqueeze(0)

    # def on_train_epoch_start(self, trainer, pl_module):
    #     if trainer.global_rank != 0:
    #         return  # 只在主进程执行
    #     print("Epoch is starting")
    #     fig1, fig2 = pl_module.plot_sample(
    #         self.prev_inputs.to(pl_module.device),
    #         self.target_residuals.to(pl_module.device)
    #     )
    #
    #     # 使用 TensorBoard 记录图像
    #     trainer.logger.experiment.add_figure('sst', fig1, global_step=trainer.global_step)
    #     trainer.logger.experiment.add_figure('thetao0', fig2, global_step=trainer.global_step)
    #     print("Uploaded samples")
    #     plt.close(fig1)
    #     plt.close(fig2)

class BackwardTimingCallback(Callback):
    def __init__(self):
        self._recent_times = []   # 存最近10次的耗时
        self._count = 0           # 反向传播计数器
        self._opt_start = None
        self._recent_optimizer_times = []

    def on_before_backward(self, trainer, pl_module, loss):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def on_after_backward(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self._recent_times.append(elapsed)
        self._count += 1

        if self._count % 10 == 0:
            avg = sum(self._recent_times) / len(self._recent_times)
            print(f"[第 {self._count} 次反向传播] 最近10次平均耗时: {avg*1000:.2f} ms")
            self._recent_times = []   # 重置，只统计下一个10次窗口

    # 新增：参数更新计时
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._opt_start = time.perf_counter()

    #  用 on_before_zero_grad 替代不存在的 on_after_optimizer_step
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if self._opt_start is None:   # ✅ 防止未赋值时就被调用
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._opt_start
        self._opt_start = None   # ✅ 用完重置，避免重复计入
        self._recent_optimizer_times.append(elapsed)

        if len(self._recent_optimizer_times) % 10 == 0:
            avg = sum(self._recent_optimizer_times) / len(self._recent_optimizer_times)
            print(f"[参数更新] 最近10次平均耗时: {avg*1000:.2f} ms")
            self._recent_optimizer_times = []

def collate_fn(batch):
    # 过滤掉 None（无效数据对）
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        # 如果整个 batch 都无效，返回一个空 batch
        return None

    # 将 prev_inputs, target_residuals 和 index 分离
    prev_inputs = torch.from_numpy(np.stack([item[0] for item in batch], axis=0))
    target_residuals = torch.from_numpy(np.stack([item[1] for item in batch], axis=0))
    indices = [item[2] for item in batch]

    return prev_inputs, target_residuals, indices

def nowcastinfer():
    input_features_dim = 93
    output_features_dim = 93

    # 初始化模型
    model = FGN(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        fine_lon=fine_lon,
        fine_lat=fine_lat,
        input_features_dim=input_features_dim,
        output_features_dim=output_features_dim,
        device='cuda',
        noise_dimension=32,
        **CFG
    )

    checkpoint_path = f"./checkpoints/mercator0.25total/lead5/epoch=0-val_loss=0.19.ckpt"  # 替换为实际的checkpoint路径
    # checkpoint_path = f"/hpcfs/fhome/wangxiang/wuwei/fgn/checkpoints/lead{LEAD}/epoch=8-val_loss=0.27.ckpt"

    state_dict = torch.load(checkpoint_path)['state_dict']
    fixed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            fixed_state_dict[k[6:]] = v  # 去掉前6个字符 ("model.")
        else:
            fixed_state_dict[k] = v  # 保留无前缀的键
    model.load_state_dict(fixed_state_dict)

    # 将模型移动到GPU并设置为评估模式
    model = model.cuda()
    model = model.float()
    model.eval()

    def normal(inputs):
        mean = np.load('./data/normalize_mean_23_0.25deg.npz')
        std = np.load('./data/normalize_std_23_0.25deg.npz')
        for i in range(96):
            inputs[0, i, :, :] = (inputs[0, i, :, :] - mean[mean.files[i]]) / std[mean.files[i]]
        return inputs

    def denormal(inputs):
        mean = np.load('./data/normalize_mean_23_0.25deg.npz')
        std = np.load('./data/normalize_std_23_0.25deg.npz')
        for i in range(93):
            if i == 0:
                inputs[0, i, :, :] = inputs[0, i, :, :].detach() * std[mean.files[i]] + mean[mean.files[i]]
            if i >= 1:
                inputs[0, i, :, :] = inputs[0, i, :, :].detach() * std[mean.files[i + 3]] + mean[mean.files[i + 3]]
        return inputs

    with torch.no_grad():
        for root, dir, files in os.walk('./data/nowcastinput'):
            for file in files:
                former_file = np.load(os.path.join(root, file))
                # mask2 = np.where(np.delete(np.isnan(former_file), [1,2,3], axis=1), 0, 1)
                former_file = normal(former_file)
                # print(np.nanmean(former_file, axis=(2,3)))
                # print(np.nanmax(former_file, axis=(2,3)))
                # print(np.nanmin(former_file, axis=(2,3)))
                former_file = np.delete(former_file, [1, 2, 3], 1)
                former_file = np.nan_to_num(former_file, nan=0)
                # former_file = torch.from_numpy(former_file).to(torch.bfloat16)
                former_file = torch.from_numpy(former_file)
                former_file = former_file * mask
                # former_file = torch.permute(former_file, (0, 3, 2, 1)).to('cuda').to(torch.bfloat16)
                former_file = torch.permute(former_file, (0, 3, 2, 1)).to('cuda')
                result = model(former_file, 1)
                result = result.mean(dim=1)
                result = result + former_file.to('cuda')
                result = torch.permute(result, (0, 3, 2, 1)).to('cpu').float()
                result = (denormal(result) * mask)[:, :, :, :].detach().numpy()
                np.save(f'./result/nowcast/lead{LEAD}/{(datetime.datetime.strptime(file[:8],"%Y%m%d")+datetime.timedelta(days=LEAD)).strftime("%Y%m%d")}.npy', result.astype(np.float16))
                print(file[:8], flush=True)

if __name__ == "__main__":
    # define dataloader
    dataset = TrainDataset(lead=LEAD)
    valdataset = ValDataset(lead=LEAD)

    dataloader = DataLoader(
        dataset,
        # shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        # drop_last=True
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        # multiprocessing_context="forkserver",
    )

    valdataloader = DataLoader(
        valdataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        # drop_last=True
    )

    # ipdb.set_trace()

    # define/resume model
    num_steps = NUM_EPOCHS * len(dataloader) // (NUM_DEVICES * NUM_ACC_GRAD)
    initial_lr = INITIAL_LR

    oceandata = xr.open_dataset('./data/19930101.nc')
    grid_lon = oceandata['lon'].values.astype(np.float32)
    grid_lat = oceandata['lat'].values
    finedata = xr.open_dataset('./data/mercatorglorys12v1_gl12_mean_1993_2016_01.nc')
    fine_lon = finedata['longitude'].values[::3]
    fine_lat = finedata['latitude'].values[::3]
    input_features_dim = 93
    output_features_dim = 93

    # checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    # print(checkpoint['optimizer_states'][0]['state'])
    #
    # denoiser = LitModel.load_from_checkpoint(
    #     checkpoint_path=CHECKPOINT_PATH,
    #     warmup=checkpoint['lr_schedulers'][0]['warmup'],
    #     learning_rate=checkpoint['lr_schedulers'][0]['_last_lr'][0],
    #     cosine_t_max=num_steps,
    #     grid_lon=grid_lon,
    #     grid_lat=grid_lat,
    #     fine_lon=fine_lon,
    #     fine_lat=fine_lat,
    #     input_features_dim=input_features_dim,
    #     output_features_dim=output_features_dim,
    #     **CFG,
    # )
    denoiser = LitModel(
        # checkpoint_path=CHECKPOINT_PATH,
        warmup=WARMUP,
        learning_rate=initial_lr,
        cosine_t_max=num_steps,
        # pressure_levels=dataset.pressure_levels,
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        fine_lon=fine_lon,
        fine_lat=fine_lat,
        input_features_dim=input_features_dim,
        output_features_dim=output_features_dim,
        **CFG,
    )
    # # denoiser = torch.compile(denoiser)
    '''
    # ── FLOPs 统计（插入此处）────────────────────────────────────────────
    from torch.profiler import profile, record_function, ProfilerActivity

    _model = denoiser.model.cuda().bfloat16().eval()
    _dummy = torch.randn(1, 681, 1440, 93, dtype=torch.bfloat16).cuda()  # (B, lat, lon, C)

    with torch.no_grad():
        _model(_dummy, 1)  # 预热

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True, record_shapes=True) as _prof:
            with record_function("model_inference"):
                _model(_dummy, 1)

    print(_prof.key_averages().table(sort_by="flops", row_limit=20))
    print(f"Total FLOPs: {sum(e.flops for e in _prof.key_averages()) / 1e9:.3f} GFLOPs")
    del _model, _dummy, _prof
    # ─────────────────────────────────────────────────────────────────────
    '''

    # define trainer
    tensorboard_logger = TensorBoardLogger(
        save_dir="./lightning_logs",
        name="mercator0.25total",
        default_hp_metric=False
    )

    checkpoint_callback = ModelCheckpoint(
        # dirpath="checkpoints/addmesh/",
        dirpath=f"./checkpoints/mercator0.25total/lead{LEAD}",
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=-1,
        # verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    timing_cb = BackwardTimingCallback()

    # 在 trainer 中使用新的 logger
    # trainer = L.Trainer(
    #     accumulate_grad_batches=NUM_ACC_GRAD,
    #     accelerator="gpu",
    #     devices=NUM_DEVICES,
    #     strategy="ddp_find_unused_parameters_true",
    #     max_epochs=NUM_EPOCHS,
    #     logger=tensorboard_logger,  # 使用 tensorboard logger
    #     callbacks=[checkpoint_callback, SamplingCallback(data=dataset[0]), lr_monitor],
    #     # callbacks=[best_checkpoint, last_checkpoint, SamplingCallback(data=dataset[0]), lr_monitor],
    #     log_every_n_steps=1,
    #     gradient_clip_val=0.5, # 添加梯度裁剪
    #     precision='bf16-mixed',
    # )

    # start training
    # print("Starting training")
    # trainer.fit(model=denoiser, train_dataloaders=dataloader, val_dataloaders=valdataloader)
    # trainer.fit(model=denoiser, train_dataloaders=dataloader, val_dataloaders=valdataloader, ckpt_path=CHECKPOINT_PATH)
    # infer()
    nowcastinfer()
    # extract_and_save_model_params('./checkpoints/mercator0.25total/addfourier/lead1/epoch=1-val_loss=0.04.ckpt',
    #                               './checkpoints/mercator0.25total/addfourier/lead1/params.ckpt')