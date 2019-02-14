

import argparse
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab, set_gpu
from bilm.data import LMDataset, BidirectionalLMDataset, BidirectionalPolyglotLMDataset

import IPython as ipy

def main(args):

    if args.gpu is not None:
        if ',' in args.gpu:
            args.gpu = args.gpu.split(',')
        n_gpus = len(args.gpu)
        set_gpu(args.gpu)
    else:
        n_gpus = 0

    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    if 'polyglot' in options or args.polyglot:
        polyglot = True
    vocab = load_vocab(args.vocab_files, max_word_length=max_word_length, polyglot=polyglot)

    prefix = args.train_prefix

    kwargs = {
        'test': False,
        'shuffle_on_load': True,
    }

    if options.get('bidirectional'):
        if 'polyglot' in options or args.polyglot:
            data = BidirectionalPolyglotLMDataset(prefix, vocab, **kwargs)
        else:
            data = BidirectionalLMDataset(prefix, vocab, **kwargs)
    else:
        data = LMDataset(prefix, vocab, **kwargs)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    # set optional inputs
    if args.n_train_tokens > 0:
        options['n_train_tokens'] = args.n_train_tokens
    if args.n_epochs > 0:
        options['n_epochs'] = args.n_epochs
    if args.batch_size > 0:
        options['batch_size'] = args.batch_size

    train(options, data, None, args.n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=ckpt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_files', nargs='+', help='Vocabulary files')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--n_train_tokens', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=0)
    parser.add_argument('--gpu', nargs='+', help='GPU id(s)')
    parser.add_argument('--polyglot', action='store_true', help='use polyglot LM even if not specified in options (e.g. due to outdated code)')

    args = parser.parse_args()
    main(args)

