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
        'compress_space',
        'residual',
        'compress_space',
        'residual',
        'compress_space',
        'residual',
        'residual',
        'residual',

    ),
    channel_multiplier=[1, 2, 4, 8],
    input_conv_kernel_size=(3, 3, 3),
    use_gan = False,
    quantizer_aux_loss_weight=1e-3,
    # lfq_entropy_loss_weight = 1,
    # lfq_commitment_loss_weight = 1000.,
    # lfq_diversity_gamma = 1,
    perceptual_loss_weight=0.00,
)

trainer = VideoTokenizerTrainer(
    tokenizer,
    #dataset_folder = '/ccn2/dataset/kinetics400/Kinetics400/k400/train/',     # folder of either videos or images, depending on setting below
    dataset_folder = '/ccn2/u/honglinc/datasets/imagenet/train',
    valid_dataset_folder='/ccn2/u/honglinc/datasets/imagenet/val',
    #dataset_folder = '/ccn2/u/honglinc/datasets/bridge_v2/demo/bridge_data_v1/berkeley/toykitchen2/close_fridge/2022-04-22_12-49-14/raw/traj_group0/traj16',
    dataset_type = 'images',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 96,
    grad_accum_every = 1,
    learning_rate = 2e-5,
    num_train_steps = 100_000,
    warmup_steps = 1000,
    use_wandb_tracking = False,
    valid_frac=0.0,
    checkpoint_every_step = 1000,
    exp_name = 'imagenet_quantize_v3_aux1e-3_lr2e-5'
)

trainer.train()

# after a lot of training ...
# can use the EMA of the tokenizer

ema_tokenizer = trainer.ema_tokenizer

# mock video

video = torch.randn(1, 3, 17, 128, 128)

# tokenizing video to discrete codes

codes = ema_tokenizer.tokenize(video) # (1, 9, 16, 16) <- in this example, time downsampled by 4x and space downsampled by 8x. flatten token ids for (non)-autoregressive training

# sanity check

decoded_video = ema_tokenizer.decode_from_code_indices(codes)

assert torch.allclose(
    decoded_video,
    ema_tokenizer(video, return_recon = True)
)