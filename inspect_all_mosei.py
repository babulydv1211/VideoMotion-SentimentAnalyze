import h5py
from pathlib import Path

root = Path('dataset', 'CMU-MOSEI')
files = [
    root / 'labels' / 'CMU_MOSEI_Labels.csd',
    root / 'visuals' / 'CMU_MOSEI_VisualFacet42.csd',
    root / 'visuals' / 'CMU_MOSEI_VisualOpenFace2.csd',
    root / 'acoustics' / 'CMU_MOSEI_COVAREP.csd',
    root / 'languages' / 'CMU_MOSEI_TimestampedWords.csd',
    root / 'languages' / 'CMU_MOSEI_TimestampedWordVectors.csd'
]

for path in files:
    if not path.exists():
        print('MISSING', path)
        continue
    print('='*80)
    print('FILE', path)
    with h5py.File(path, 'r') as f:
        print('top keys', list(f.keys())[:20])
        print('attrs', dict(f.attrs))
        for k in list(f.keys())[:5]:
            obj = f[k]
            print('  key:', k, type(obj).__name__, getattr(obj, 'shape', None), getattr(obj, 'dtype', None))
            if isinstance(obj, h5py.Group):
                print('   subkeys', list(obj.keys())[:20])
                if 'data' in obj:
                    data = obj['data']
                    sample = list(data.keys())[0]
                    print('   sample id', sample)
                    sg = data[sample]
                    print('    sample keys', list(sg.keys()))
                    for kk in list(sg.keys())[:5]:
                        o = sg[kk]
                        print('      ', kk, type(o).__name__, getattr(o, 'shape', None), getattr(o, 'dtype', None))
                        if hasattr(o, 'shape'):
                            try:
                                print('        sample values', o[:2])
                            except Exception as e:
                                print('        sample err', e)
                else:
                    print('   no data group')
