"""
Generate a list of sorted labels of the selected attribute from a list of filelists
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--filelists', nargs='+', type=str, required=True,
                    help='paths to the filelists')
parser.add_argument('--attr_name', type=str, required=True,
                    help='attribute name to extract from the filelist')
parser.add_argument('--attr_list', type=str, required=True,
                    help='output list')
args = parser.parse_args()
print(args)

attrs = set()
for filelist in args.filelists:
    with open(filelist, 'r') as f:
        for line in f:
            datum = json.loads(line.rstrip())
            try:
                attrs.add(datum[args.attr_name])
            except KeyError as e:
                raise(e)

attrs = sorted(attrs)
print('total %d distinct values of %s' % (len(attrs), args.attr_name))
print('  %s...' % str(attrs[:5]))

assert(not os.path.exists(args.attr_list))
with open(args.attr_list, 'w') as f:
    for attr in attrs:
        f.write('%s\n' % attr)
print('dumped attribute value list to %s' % args.attr_list)
