import argparse
import re
from text import symbols

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    ################################
    # Experiment Parameters        #
    ################################
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--iters_per_checkpoint', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dynamic_loss_scaling', type=str2bool, default=True)
    parser.add_argument('--fp16_run', type=str2bool, default=False)
    parser.add_argument('--distributed_run', type=str2bool, default=False)
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:54321')
    parser.add_argument('--cudnn_enabled', type=str2bool, default=True)
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=False)
    parser.add_argument('--ignore_layers', type=str, nargs='*', 
                        default=['embedding.weight'])
    
    ################################
    # Data Parameters             #
    ################################
    parser.add_argument('--load_mel_from_disk', type=str2bool, default=False)
    parser.add_argument('--training_files', type=str, 
                        default='filelists/ljs_audio_text_train_filelist.txt')
    parser.add_argument('--validation_files', type=str, 
                        default='filelists/ljs_audio_text_val_filelist.txt')
    parser.add_argument('--text_cleaners', type=str, nargs='*', 
                        default=['english_cleaners'])
    parser.add_argument('--text_or_code', type=str, default='text')
    parser.add_argument('--code_key', type=str, default='')
    parser.add_argument('--code_dict', type=str, default='')
    parser.add_argument('--collapse_code', type=str2bool, default=True)

    ################################
    # Audio Parameters             #
    ################################
    parser.add_argument('--max_wav_value', type=float, default=32768.0)
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--filter_length', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=256)
    parser.add_argument('--win_length', type=int, default=1024)
    parser.add_argument('--n_mel_channels', type=int, default=80)
    parser.add_argument('--mel_fmin', type=float, default=0.0)
    parser.add_argument('--mel_fmax', type=float, default=8000.0)
    
    ################################
    # Model Parameters             #
    ################################
    parser.add_argument('--n_symbols', type=int, default=len(symbols))
    parser.add_argument('--symbols_embedding_dim', type=int, default=512)
    parser.add_argument('--symbols_embedding_path', type=str, default='')
    
    # Encoder parameters
    parser.add_argument('--encoder_kernel_size', type=int, default=5)
    parser.add_argument('--encoder_n_convolutions', type=int, default=3)
    parser.add_argument('--encoder_embedding_dim', type=int, default=512)
    
    # Decoder parameters
    parser.add_argument('--n_frames_per_step', type=int, default=1)
    parser.add_argument('--decoder_rnn_dim', type=int, default=1024)
    parser.add_argument('--prenet_dim', type=int, default=256)
    parser.add_argument('--max_decoder_steps', type=int, default=1000)
    parser.add_argument('--gate_threshold', type=float, default=0.5)
    parser.add_argument('--p_attention_dropout', type=float, default=0.1)
    parser.add_argument('--p_decoder_dropout', type=float, default=0.1)

    # Attention parameters
    parser.add_argument('--attention_rnn_dim', type=int, default=1024)
    parser.add_argument('--attention_dim', type=int, default=128)
    
    # Location Layer parameters
    parser.add_argument('--attention_location_n_filters', type=int, default=32)
    parser.add_argument('--attention_location_kernel_size', type=int, default=31)
    
    # Mel-post processing network parameters
    parser.add_argument('--postnet_embedding_dim', type=int, default=512)
    parser.add_argument('--postnet_kernel_size', type=int, default=5)
    parser.add_argument('--postnet_n_convolutions', type=int, default=5)

    ################################
    # Optimization Hyperparameters #
    ################################
    parser.add_argument('--use_saved_learning_rate', type=str2bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip_thresh', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mask_padding', type=str2bool, default=True)
    return parser

def create_hparams(hparams_string='', verbose=False):
    parser = create_parser()
    hparams_string = re.sub(r'([^=,]*=)', r'--\1', hparams_string)
    tokens = hparams_string.replace(',', ' ').replace('=', ' ').split()
    argv = [v for v in tokens if bool(v)]
    args = parser.parse_args(argv)
    
    if hparams_string:
        print('Parsing command line hparams: %s' % hparams_string)

    if verbose:
        print('Final parsed hparams:')
        for k in vars(args):
            print('%-40s : %s' % (k, getattr(args, k)))

    return args
