from text import symbols
import argparse

defaults = {
    ################################
    # Experiment Parameters        #
    ################################
    'epochs':500,
    'iters_per_checkpoint':10000,
    'seed':1234,
    'dynamic_loss_scaling':True,
    'fp16_run':False,
    'distributed_run':False,
    'dist_backend':"nccl",
    'dist_url':"tcp://localhost:54321",
    'cudnn_enabled':True,
    'cudnn_benchmark':False,
    'ignore_layers':"['embedding.weight']",

    ################################
    # Data Parameters             #
    ################################
    'load_mel_from_disk':False,
    'training_files':'filelists/train_filelist.txt',
    'validation_files':'filelists/val_filelist.txt',
    'text_cleaners':"['korean_cleaners']",  ##한글을 사용하려면 "['korean_cleaners']"   -> str 형태로 입력해야 함
                                            ##영어를 사용하려면 "['english_cleaners']"
    ################################
    # Audio Parameters             #
    ################################
    'max_wav_value':32768.0,
    'sampling_rate':22050,
    'filter_length':1024,
    'hop_length':256,
    'win_length':1024,
    'n_mel_channels':80,
    'mel_fmin':0.0,
    'mel_fmax':8000.0,

    ################################
    # Model Parameters             #
    ################################
    'n_symbols':len(symbols),
    'symbols_embedding_dim':512,

    # Encoder parameters
    'encoder_kernel_size':5,
    'encoder_n_convolutions':3,
    'encoder_embedding_dim':512,

    # Decoder parameters
    'n_frames_per_step':1,  # currently only 1 is supported
    'decoder_rnn_dim':1024,
    'prenet_dim':256,
    'max_decoder_steps':1000,
    'gate_threshold':0.5,
    'p_attention_dropout':0.1,
    'p_decoder_dropout':0.1,

    # Attention parameters
    'attention_rnn_dim':1024,
    'attention_dim':128,

    # Location Layer parameters
    'attention_location_n_filters':32,
    'attention_location_kernel_size':31,

    # Mel-post processing network parameters
    'postnet_embedding_dim':512,
    'postnet_kernel_size':5,
    'postnet_n_convolutions':5,

    ################################
    # Optimization Hyperparameters #
    ################################
    'use_saved_learning_rate':False,
    'learning_rate':1e-3,
    'weight_decay':1e-6,
    'grad_clip_thresh':1.0,
    'batch_size':32,
    'gradient_accumulation_steps':2,
    'mask_padding':True  # set model's padded outputs to padded values
}

def get_hparams(args, parser):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    ###################
    hparams = arg_groups['hparams']
    hparams.text_cleaners = eval(hparams.text_cleaners)
    hparams.ignore_layers = eval(hparams.ignore_layers)

    return hparams

def add_hparams(parser):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams_group = parser.add_argument_group('hparams')
    for key, default in defaults.items():
        hparams_group.add_argument("--{}".format(key), type=type(default), default=default)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams_group = parser.add_argument_group('hparams')
    for key, default in defaults.items():
        hparams_group.add_argument("--{}".format(key), type=type(default), default=default)
    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    ###################
    hparams = arg_groups['hparams']

    print(hparams)
    print(type(hparams.text_cleaners))
    print(type(eval(hparams.text_cleaners)))
    print("############################")
    print(args)
