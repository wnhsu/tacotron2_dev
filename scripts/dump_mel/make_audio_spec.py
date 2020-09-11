import argparse
import json
import os

def get_tar_path(src_path, out_dir, nlevels):
    """
    suppose the src_path is <dir1>/.../<dirN-1>/<dirN>/<basename>.wav
    the tar_path will be <out_dir>/<basename>.npz if nlevels == 1, and
    <out_dir>/<dirN>/<basename>.npz if nlevels == 2
    """
    basename = os.path.splitext(os.path.basename(src_path))[0]
    if nlevels == 1:
        dirs = []
    else:
        ndirs = nlevels - 1
        dirs = os.path.dirname(src_path).split('/')
        assert(len(dirs) >= ndirs)
        dirs = dirs[-ndirs:]
    return '/'.join([out_dir] + dirs + [basename]) + '.npy'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cap_filelist', type=str)
    parser.add_argument('feat_out_dir', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--nlevels', type=int, default=1)
    args = parser.parse_args()
    
    with open(args.cap_filelist, 'r') as f:
        data = [json.loads(line.rstrip()) for line in f]
    print('total %d audio files' % len(data))
    
    if os.path.exists(args.out_path):
        raise ValueError('%s already exists. exiting...' % (args.out_path))
    
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        for datum in data:
            src_path = datum['audio']
            tar_path = get_tar_path(src_path, args.feat_out_dir, args.nlevels)

            f.write('%s %s\n' % (src_path, tar_path))
