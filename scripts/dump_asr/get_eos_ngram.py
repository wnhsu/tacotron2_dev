import argparse
import sys
sys.path.append('/usr/users/wnhsu/code/tacotron2_factory/tacotron2_20191017_dev/')
from collections import Counter

from text import code_to_sequence
from utils import load_code_dict, load_filepaths_and_text

parser = argparse.ArgumentParser()
parser.add_argument('--filelist', type=str, required=True)
parser.add_argument('--code_dict', type=str, required=True)
parser.add_argument('--code_key', type=str, required=True)
parser.add_argument('--ngram', type=int, default=2)
args = parser.parse_args()

data = load_filepaths_and_text(args.filelist)
code_dict = load_code_dict(args.code_dict, False, False)
inv_code_dict = {v: k for k, v in code_dict.items()}

eos_ngram_counter = Counter()
for d in data:
    seq = code_to_sequence(d[args.code_key].split(),
                           code_dict, collapse_code=True)
    eos_ngram = tuple(seq[-args.ngram:])
    eos_ngram_counter[eos_ngram] += 1

num_utts = len(data)
print('Summary over %d sequences' % num_utts)
for ngram, count in eos_ngram_counter.most_common(20):
    code_ngram = [inv_code_dict[x] for x in ngram]
    print('%-30s : %6d (%4.2f)' % (code_ngram, count, count / num_utts))
