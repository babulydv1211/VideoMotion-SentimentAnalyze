import h5py
from pathlib import Path

root = Path(r'..\datasets\CMU-MOSEI')
labels = root / 'labels' / 'CMU_MOSEI_Labels.csd'
print('LABEL FILE', labels)
with h5py.File(labels, 'r') as f:
    keys = list(f.keys())
    print('top keys', keys[:20])
    print('attrs', dict(f.attrs))
    for k in keys[:10]:
        obj = f[k]
        print('key', k, 'type', type(obj).__name__, 'shape', getattr(obj, 'shape', None), 'dtype', getattr(obj, 'dtype', None))
        if isinstance(obj, h5py.Group):
            subkeys = list(obj.keys())
            print('  subkeys', subkeys[:20])
            for sk in subkeys[:10]:
                sobj = obj[sk]
                print('   subkey', sk, 'type', type(sobj).__name__, 'shape', getattr(sobj, 'shape', None), 'dtype', getattr(sobj, 'dtype', None))
    if keys:
        group = f[keys[0]]
        print('group size', len(group))
        data_group = group['data']
        print('data_count', len(data_group))
        first_vid = list(data_group.keys())[0]
        print('first video id', first_vid)
        vid_group = data_group[first_vid]
        print('  vid_group keys', list(vid_group.keys()))
        for leaf in vid_group.keys():
            leafobj = vid_group[leaf]
            print('   leaf', leaf, 'type', type(leafobj).__name__, 'shape', getattr(leafobj, 'shape', None), 'dtype', getattr(leafobj, 'dtype', None))
            if isinstance(leafobj, h5py.Dataset):
                try:
                    print('     sample', leafobj[:5])
                except Exception as e:
                    print('     sample error', e)
        print('model label names maybe in metadata:')
        print('dimension_names', group['metadata']['dimension names'][:])
