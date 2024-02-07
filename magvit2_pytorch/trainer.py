from pathlib import Path
from functools import partial
from contextlib import contextmanager, nullcontext
import os
import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, random_split
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
import pytorch_warmup as warmup
from glob import glob
from beartype import beartype
from transformers import get_scheduler
from beartype.typing import Optional, Literal, Union, Type

from magvit2_pytorch.optimizer import get_optimizer

from magvit2_pytorch.magvit2_pytorch import VideoTokenizer

from magvit2_pytorch.data import (
    VideoDataset,
    ImageDataset,
    DataLoader,
    video_tensor_to_gif
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from einops import rearrange
import lpips
from ema_pytorch import EMA

from pytorch_custom_utils import auto_unwrap_model

# constants

VideosOrImagesLiteral = Union[
    Literal['videos'],
    Literal['images']
]

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = True
)

# helpers

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data
# class

@auto_unwrap_model()
class VideoTokenizerTrainer:
    @beartype
    def __init__(
        self,
        model: VideoTokenizer,
        *,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float = 1e-5,
        grad_accum_every: int = 1,
        apply_gradient_penalty_every: int = 4,
        max_grad_norm: Optional[float] = None,
        dataset: Optional[Dataset] = None,
        dataset_folder: Optional[str] = None,
        valid_dataset_folder: Optional[str] = None,
        dataset_type: VideosOrImagesLiteral = 'videos',
        checkpoints_folder = './checkpoints',
        results_folder = './results',
        exp_name = 'exp',
        random_split_seed = 42,
        valid_frac = 0.05,
        validate_every_step = 100,
        checkpoint_every_step = 100,
        num_frames = 17,
        use_wandb_tracking = False,
        discr_start_after_step = 0.,
        warmup_steps = 1000,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        dataset_kwargs: dict = dict(),
        auto_resume: bool = True,
    ):
        exp_name = f'{exp_name}_bs{batch_size}_ac{grad_accum_every}'
        self.use_wandb_tracking = use_wandb_tracking

        if use_wandb_tracking:
            accelerate_kwargs['log_with'] = 'wandb'


        if 'kwargs_handlers' not in accelerate_kwargs:
            accelerate_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        # instantiate accelerator

        self.accelerator = Accelerator(**accelerate_kwargs)

        if use_wandb_tracking:
            self.accelerator.init_trackers(project_name="magvit2", init_kwargs={"wandb":{"name":exp_name}})

        # model and exponentially moving averaged model

        self.model = model

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f'Total Parameters: {total_params:.2f}M')

        if self.is_main:
            self.ema_model = EMA(
                model,
                include_online_model = False,
                **ema_kwargs
            )

        dataset_kwargs.update(channels = model.channels)

        # dataset

        if not exists(dataset):
            if dataset_type == 'videos':
                dataset_klass = VideoDataset
                dataset_kwargs = {**dataset_kwargs, 'num_frames': num_frames}
            else:
                dataset_klass = ImageDataset

            assert exists(dataset_folder)
            dataset = dataset_klass(dataset_folder, image_size = model.image_size, **dataset_kwargs)

        # splitting dataset for validation

        assert 0 <= valid_frac < 1.

        if valid_dataset_folder is None:
            # train_size = int((1 - valid_frac) * len(dataset))
            # valid_size = len(dataset) - train_size
            train_size = len(dataset) - 8
            valid_size = 8
            dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))

            self.print(f'training with dataset of {len(dataset)} samples and validating with randomly splitted {len(valid_dataset)} samples')
        else:
            valid_dataset = dataset_klass(valid_dataset_folder, image_size = model.image_size, **dataset_kwargs)
            self.print(f'training valid dataset of {len(dataset)} samples')

        # dataset and dataloader

        self.dataset = dataset
        self.dataloader = DataLoader(dataset, shuffle = True, drop_last = True, batch_size = batch_size, num_workers = 4)

        self.valid_dataset = valid_dataset
        self.valid_dataloader = DataLoader(valid_dataset, shuffle = False, drop_last = True, batch_size = batch_size, num_workers = 4)

        self.validate_every_step = validate_every_step
        self.checkpoint_every_step = checkpoint_every_step

        # optimizerss
        self.optimizer = get_optimizer(model.parameters(), lr = learning_rate, **optimizer_kwargs)
        self.discr_optimizer = get_optimizer(model.discr_parameters(), lr = learning_rate, **optimizer_kwargs)

        # learning rate scheduler

        self.scheduler = get_scheduler(
            name='linear',
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps * self.accelerator.num_processes,
            num_training_steps=num_train_steps * self.accelerator.num_processes,
        )

        self.discr_scheduler = get_scheduler(
            name='linear',
            optimizer=self.discr_optimizer,
            num_warmup_steps=warmup_steps * self.accelerator.num_processes,
            num_training_steps=num_train_steps * self.accelerator.num_processes,
        )

        # training related params

        self.batch_size = batch_size

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm

        self.apply_gradient_penalty_every = apply_gradient_penalty_every

        # prepare for maybe distributed
        (
            self.model,
            self.dataloader,
            self.valid_dataloader,
            self.optimizer,
            self.discr_optimizer,
            self.scheduler,
            self.discr_scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.dataloader,
            self.valid_dataloader,
            self.optimizer,
            self.discr_optimizer,
            self.scheduler,
            self.discr_scheduler
        )

        # only use adversarial training after a certain number of steps

        self.discr_start_after_step = discr_start_after_step

        # multiscale discr losses

        try:
            self.has_multiscale_discrs = self.model.module.has_multiscale_discrs
            self.multiscale_discrs = self.model.module.multiscale_discrs
        except:
            self.has_multiscale_discrs = self.model.has_multiscale_discrs
            self.multiscale_discrs = self.model.multiscale_discrs

        self.multiscale_discr_optimizers = []

        for ind, discr in enumerate(self.multiscale_discrs):
            multiscale_optimizer = get_optimizer(discr.parameters(), lr = learning_rate, **optimizer_kwargs)

            self.multiscale_discr_optimizers.append(multiscale_optimizer)

        if self.has_multiscale_discrs:
            self.multiscale_discr_optimizers = self.accelerator.prepare(*self.multiscale_discr_optimizers)

        # checkpoints and sampled results folder

        checkpoints_folder = Path(os.path.join(checkpoints_folder, exp_name))
        results_folder = Path(os.path.join(results_folder, exp_name))

        checkpoints_folder.mkdir(parents = True, exist_ok = True)
        results_folder.mkdir(parents = True, exist_ok = True)

        assert checkpoints_folder.is_dir()
        assert results_folder.is_dir()

        self.checkpoints_folder = checkpoints_folder
        self.results_folder = results_folder

        # keep track of train step

        self.step = 0

        # move ema to the proper device
        if self.is_main:
            self.ema_model.to(self.device)

        if auto_resume:
            latest_checkpoint = max(glob(os.path.join(checkpoints_folder, '*.pt')), key = os.path.getctime, default = None)

            if latest_checkpoint is not None:
                print('Auto-resume from latest checkpoint:', latest_checkpoint)
                self.load(latest_checkpoint)

        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
    @contextmanager
    @beartype
    def trackers(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        hps: Optional[dict] = None
    ):
        assert self.use_wandb_tracking

        self.accelerator.init_trackers(project_name, config = hps)

        if exists(run_name):
            self.accelerator.trackers[0].run.name = run_name

        yield
        self.accelerator.end_training()

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step)

    @property
    def device(self):
        return self.model.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.model.state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
            discr_optimizer = self.discr_optimizer.state_dict(),
            # warmup = self.warmup.state_dict(),
            scheduler = self.scheduler.state_dict(),
            # discr_warmup = self.discr_warmup.state_dict(),
            discr_scheduler = self.discr_scheduler.state_dict(),
            step = self.step
        )

        for ind, opt in enumerate(self.multiscale_discr_optimizers):
            pkg[f'multiscale_discr_optimizer_{ind}'] = opt.state_dict()

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location='cpu')

        self.model.load_state_dict(pkg['model'])
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.discr_optimizer.load_state_dict(pkg['discr_optimizer'])
        # self.warmup.load_state_dict(pkg['warmup'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        # self.discr_warmup.load_state_dict(pkg['discr_warmup'])
        self.discr_scheduler.load_state_dict(pkg['discr_scheduler'])

        if self.is_main:
            self.ema_model.load_state_dict(pkg['ema_model'])

        for ind, opt in enumerate(self.multiscale_discr_optimizers):
            opt.load_state_dict(pkg[f'multiscale_discr_optimizer_{ind}'])

        self.step = pkg['step']

    def train_step(self, dl_iter):
        self.model.train()

        step = self.step

        # determine whether to train adversarially

        train_adversarially = self.model.use_gan and (step + 1) > self.discr_start_after_step

        adversarial_loss_weight = 0. if not train_adversarially else None
        multiscale_adversarial_loss_weight = 0. if not train_adversarially else None

        # main model

        self.optimizer.zero_grad()

        for grad_accum_step in range(self.grad_accum_every):

            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

            data, *_ = next(dl_iter)

            with self.accelerator.autocast(), context():
                loss, loss_breakdown = self.model(
                    data,
                    return_loss = True,
                    adversarial_loss_weight = adversarial_loss_weight,
                    multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight
                )

                self.accelerator.backward(loss / self.grad_accum_every)

        self.log(
            total_loss = loss.item(),
            recon_loss = loss_breakdown.recon_loss.item(),
            lfq_aux_loss = loss_breakdown.lfq_aux_loss.item(),
            per_sample_entropy = loss_breakdown.quantizer_loss_breakdown.per_sample_entropy.item(),
            commitment = loss_breakdown.quantizer_loss_breakdown.commitment.item(),
            batch_entropy = loss_breakdown.quantizer_loss_breakdown.batch_entropy.item(),
            perceptual_loss=loss_breakdown.perceptual_loss.item(),
            adversarial_gen_loss=loss_breakdown.adversarial_gen_loss.item(),
            lr = self.optimizer.param_groups[1]['lr']
        )



        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()


        # update ema model

        self.wait()

        if self.is_main:
            self.ema_model.update()

        self.wait()

        # if adversarial loss is turned off, continue

        if not train_adversarially:
            self.step += 1
            return

        # discriminator and multiscale discriminators

        self.discr_optimizer.zero_grad()

        if self.has_multiscale_discrs:
            for multiscale_discr_optimizer in self.multiscale_discr_optimizers:
                multiscale_discr_optimizer.zero_grad()

        apply_gradient_penalty = not (step % self.apply_gradient_penalty_every)

        for grad_accum_step in range(self.grad_accum_every):

            is_last = grad_accum_step == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

            data, *_ = next(dl_iter)

            with self.accelerator.autocast(), context():
                discr_loss, discr_loss_breakdown = self.model(
                    data,
                    return_discr_loss = True,
                    apply_gradient_penalty = apply_gradient_penalty
                )

                self.accelerator.backward(discr_loss / self.grad_accum_every)

        self.log(discr_loss = discr_loss_breakdown.discr_loss.item())

        if apply_gradient_penalty:
            self.log(gradient_penalty = discr_loss_breakdown.gradient_penalty.item())

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.discr_parameters(), self.max_grad_norm)

            if self.has_multiscale_discrs:
                for multiscale_discr in self.multiscale_discrs:
                    self.accelerator.clip_grad_norm_(multiscale_discr.parameters(), self.max_grad_norm)

        self.discr_optimizer.step()
        self.discr_scheduler.step()

        self.print(
            f"step: {step}, "
            f"lr: {self.optimizer.param_groups[1]['lr']:.7f}, "
            f"total: {loss.item():.5f}, "
            f"recon: {loss_breakdown.recon_loss.item():.5f}, "
            f"distr: {discr_loss_breakdown.discr_loss.item():.5f},"
            f"lfq_aux: {loss_breakdown.lfq_aux_loss.item():.5f}, "
            f"perceptual: {loss_breakdown.perceptual_loss.item():.5f}, "
            f"sample_entropy: {loss_breakdown.quantizer_loss_breakdown.per_sample_entropy.item():.5f}, "
            f"commitment: {loss_breakdown.quantizer_loss_breakdown.commitment.item():.5f}, "
            f"batch_entropy: {loss_breakdown.quantizer_loss_breakdown.batch_entropy.item():.5f}"
        )

        if self.has_multiscale_discrs:
            for multiscale_discr_optimizer in self.multiscale_discr_optimizers:
                multiscale_discr_optimizer.step()

        # update train step

        self.step += 1

    @torch.no_grad()
    def valid_step(
        self,
        dl_iter,
        save_recons = True,
        num_save_recons = 1
    ):
        # if self.is_main:
        #     self.ema_model.eval()

        recon_loss = 0.
        ema_recon_loss = 0.
        lpips = 0.
        num_steps = 0

        valid_videos = []
        recon_videos = []

        for _ in range(len(self.valid_dataloader)):

            valid_video, = next(dl_iter)
            valid_video = valid_video.to(self.device)

            with self.accelerator.autocast():
                loss, recon_video = self.model(valid_video, return_recon_loss_only = True)
                ema_loss, ema_recon_video = loss, recon_video #self.ema_model(valid_video, return_recon_loss_only = True)

            recon_loss += loss
            ema_recon_loss += ema_loss

            # Compute LPIPS


            if valid_video.ndim == 4:
                valid_video = rearrange(valid_video, 'b c h w -> b c 1 h w')

            lpips += self.lpips_fn(
                (recon_video * 2 - 1).permute(0, 2, 1, 3, 4).flatten(0, 1),
                (valid_video * 2 - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
            ).mean()

            valid_videos.append(valid_video.cpu())
            recon_videos.append(recon_video.cpu())
            num_steps += 1


        recon_loss /= num_steps
        ema_recon_loss /= num_steps
        lpips /= num_steps


        self.log(
            valid_recon_loss = recon_loss.item(),
            valid_ema_recon_loss = ema_recon_loss.item(),
            valid_lpips = lpips.item()
        )

        self.print(f'validation recon loss {recon_loss:.3f}')
        self.print(f'validation EMA recon loss {ema_recon_loss:.3f}')
        self.print(f'validation lpips {lpips:.3f}')

        if not save_recons:
            return

        valid_videos = torch.cat(valid_videos)
        recon_videos = torch.cat(recon_videos)

        recon_videos.clamp_(min = 0., max = 1.)

        valid_videos, recon_videos = map(lambda t: t[:num_save_recons], (valid_videos, recon_videos))

        real_and_recon = rearrange([valid_videos, recon_videos], 'n b c f h w -> c f (b h) (n w)')

        validate_step = self.step // self.validate_every_step

        device_id = str(self.device).split('cuda:')[-1]

        sample_path = os.path.join(self.results_folder, f'sampled.{validate_step}.{device_id}.gif')

        video_tensor_to_gif(real_and_recon, str(sample_path))

        self.print(f'sample saved to {str(sample_path)}')

    def train(self):

        step = self.step

        dl_iter = cycle(self.dataloader)
        valid_dl_iter = cycle(self.valid_dataloader)

        while step < self.num_train_steps:


            self.train_step(dl_iter)

            self.wait()

            if not (step % self.validate_every_step):
                self.valid_step(valid_dl_iter)

            # self.wait()

            if self.is_main and not ((step + 1) % self.checkpoint_every_step):
                checkpoint_num = step // self.checkpoint_every_step
                checkpoint_path = self.checkpoints_folder / f'checkpoint.{checkpoint_num}.pt'
                self.save(str(checkpoint_path))

            # self.wait()

            step += 1
