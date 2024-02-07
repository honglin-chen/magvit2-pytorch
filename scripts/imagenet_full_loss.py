from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)

tokenizer = VideoTokenizer(
    image_size = 128,
    init_dim = 64,
    max_dim = 512,
    codebook_size = 1024,
    num_codebooks = 2,
    layers = (
        'residual',
        'residual',
        'residual',
        'residual',
        'compress_space',
        'residual',
        'residual',
        'residual',
        'residual',
        'compress_space',
        'residual',
        'residual',
        'residual',
        'residual',
        'compress_space',
        'residual',
        'residual',
        'residual',
        'residual',
        'residual',
        'residual',

    ),
    channel_multiplier=[1, 2, 2, 4],
    input_conv_kernel_size=(3, 3, 3),
    use_gan = True,
    quantizer_aux_loss_weight=1e-3,
    # lfq_entropy_loss_weight = 1,
    # lfq_commitment_loss_weight = 1000.,
    # lfq_diversity_gamma = 1,
    perceptual_loss_weight=0.1,
    adversarial_loss_weight=0.1,
)

trainer = VideoTokenizerTrainer(
    tokenizer,
    dataset_folder = '/ccn2/u/honglinc/datasets/imagenet/train',
    valid_dataset_folder='/ccn2/u/honglinc/datasets/imagenet/val',
    dataset_type = 'images',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 16,
    grad_accum_every = 3,
    learning_rate = 2e-5,
    num_train_steps = 50_000,
    warmup_steps = 1000,
    use_wandb_tracking = True,
    valid_frac=0.0,
    checkpoint_every_step = 2000,
    validate_every_step=2000,
    exp_name = 'im_au1e-3_pe1e-1_ad1e-1_lr2e-5_st5e4'
)

trainer.train()
