"""Separate ordinary-training IVF/FastScan pilot; no IVF NMSE claim."""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import faiss
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rht_helpers', ROOT/'benchs/bench_rht.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

def make_index(d, family, bits, mode, seed, padded, nlist):
    d_out = 1 << (d-1).bit_length() if padded or mode.startswith('rht') else d
    transform = None
    if mode.startswith('rht'):
        transform = faiss.HadamardRotation(d, seed, int(mode[-1]))
    elif mode == 'urr':
        transform = faiss.LinearTransform(d, d_out)
        faiss.copy_array_to_vector(helpers.haar_matrix(d_out, seed)[:, :d].copy().ravel(), transform.A)
        transform.is_trained = transform.is_orthonormal = True
    elif d_out != d:
        transform = faiss.RemapDimensionsTransform(d, d_out, False)
    token = {'RaBitQ': f'RaBitQ{bits}', 'EDEN': f'EDEN{bits}',
             'RaBitQFastScan': f'RaBitQfs{bits}', 'Flat': 'Flat'}[family]
    leaf = faiss.index_factory(d_out, f'IVF{nlist},{token}', faiss.METRIC_L2)
    leaf.cp.seed = 97531
    leaf.cp.niter = 25
    leaf.cp.nredo = 1
    if family.startswith('RaBitQ'):
        leaf.qb = 4
    index = leaf if transform is None else faiss.IndexPreTransform(transform, leaf)
    return index, leaf, d_out

def storage(index, leaf):
    sizes = [int(leaf.invlists.list_size(i)) for i in range(leaf.nlist)]
    if isinstance(leaf, faiss.IndexIVFRaBitQFastScan):
        packer = leaf.get_CodePacker()
        code_bytes = sum(((n+packer.nvec-1)//packer.nvec)*packer.block_size for n in sizes)
        block = {'vectors_per_block': int(packer.nvec), 'block_size_bytes': int(packer.block_size)}
    else:
        code_bytes = sum(sizes)*leaf.code_size
        block = None
    return {'serialized_size_bytes': int(faiss.serialize_index(index).size),
            'physical_code_bytes_including_block_padding': int(code_bytes),
            'ivf_id_bytes': int(sum(sizes)*8), 'code_size_bytes': int(leaf.code_size),
            'list_sizes': sizes, 'block_storage': block}

def smoke(xt, xb, xq):
    xt, xb, xq = xt[:256], xb[:512], xq[:5]
    exact = faiss.IndexFlatL2(xb.shape[1]); exact.add(xb)
    truth_d, truth_i = exact.search(xq, 10)
    count = 0
    for padded, mode in itertools.product((False, True), helpers.MODES):
        for family in ('RaBitQ', 'EDEN', 'RaBitQFastScan', 'Flat'):
            index, leaf, _ = make_index(xb.shape[1], family, 2, mode, 12345, padded, 4)
            index.train(xt); index.add(xb); leaf.nprobe = 4
            distances, predicted = index.search(xq, 10)
            assert np.isfinite(distances).all() and (predicted >= 0).all()
            assert (predicted < len(xb)).all()
            assert all(len(set(row)) == 10 for row in predicted)
            transformed = helpers.apply_transform(faiss, index, xq)
            leaf_d, leaf_i = leaf.search(transformed, 10)
            np.testing.assert_array_equal(predicted, leaf_i)
            np.testing.assert_allclose(distances, leaf_d, rtol=1e-6, atol=1e-5)
            restored = faiss.deserialize_index(faiss.serialize_index(index))
            restore_d, restore_i = restored.search(xq, 10)
            np.testing.assert_array_equal(restore_i, predicted)
            np.testing.assert_array_equal(restore_d, distances)
            if family == 'Flat':
                np.testing.assert_array_equal(predicted, truth_i)
                np.testing.assert_allclose(distances, truth_d, rtol=2e-5, atol=1e-4)
            assert storage(index, leaf)['ivf_id_bytes'] == len(xb)*8
            count += 1
    print(f'ANN_SMOKE_OK: {count} cases; wrapper/leaf, serialization, exact IVFFlat oracle, IDs and storage', flush=True)

def run(args):
    helpers.require_host('Amumu') if hasattr(helpers, 'require_host') else None
    import platform
    assert platform.node().split('.')[0].lower() == 'amumu'
    with np.load(args.npz) as data:
        xt, xb, xq = [np.ascontiguousarray(data[k], dtype='float32') for k in ('xt','xb','xq')]
    if args.smoke:
        smoke(xt, xb, xq)
        return
    exact = faiss.IndexFlatL2(xb.shape[1]); exact.add(xb)
    _, truth = exact.search(xq, 100)
    report = {'schema_version': 1, 'status': 'running',
              'metadata': helpers.metadata(faiss, args),
              'config': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'source_npz_sha256': helpers.file_record(args.npz, True),
              'shapes': {k: list(v.shape) for k,v in [('xt',xt),('xb',xb),('xq',xq)]},
              'training': 'ordinary independent factory training per transform; fixed cp.seed=97531, niter=25, nredo=1',
              'groundtruth': 'exact IndexFlatL2 on actual original-space subset; first k of exact top100',
              'diagnostics': 'No IVF reconstruction NMSE; flat diagnostics are separate',
              'driver_version': 'v2: selected supported factories only; unexpected RuntimeError fails the run',
              'threads': threadpool_info(), 'affinity_mask': os.getenv('AMUMU_EFFECTIVE_AFFINITY_MASK'),
              'rows': [], 'unsupported': []}
    cases = list(itertools.product(('RaBitQ','EDEN','RaBitQFastScan'), (2,3,4), helpers.MODES, range(12345,12350)))
    np.random.default_rng(20261020).shuffle(cases)
    report['case_order'] = [list(c) for c in cases]
    helpers.write_results(args.output, report)
    for number, (family,bits,mode,seed) in enumerate(cases, 1):
        begin = time.perf_counter()
        index, leaf, d_out = make_index(xb.shape[1], family, bits, mode, seed, args.pad_baselines, 64)
        init = time.perf_counter()-begin
        begin = time.perf_counter(); index.train(xt); train = time.perf_counter()-begin
        begin = time.perf_counter(); index.add(xb); add = time.perf_counter()-begin
        memory = storage(index, leaf)
        for nprobe, k in itertools.product((1,2,4,8,16,32,64), (10,100)):
            leaf.nprobe = nprobe
            helpers.search_batches(index,xq,k,256)
            samples=[]
            for _ in range(3):
                begin=time.perf_counter(); predicted=helpers.search_batches(index,xq,k,256)
                samples.append(time.perf_counter()-begin)
            recall=helpers.per_query_recall(truth[:,:k],predicted)
            transforms=[]
            for _ in range(3):
                begin=time.perf_counter(); helpers.apply_transform(faiss,index,xq)
                transforms.append(time.perf_counter()-begin)
            report['rows'].append({'family':family,'bits':bits,'mode':mode,'seed':seed,
                'nlist':64,'nprobe':nprobe,'k':k,'qb':4 if family.startswith('RaBitQ') else None,
                'd_in':xb.shape[1],'d_out':d_out,'by_residual':bool(leaf.by_residual),
                'coarse_cp_seed':97531,'coarse_cp_niter':25,'coarse_cp_nredo':1,
                'max_points_per_centroid':int(leaf.cp.max_points_per_centroid),
                'init_seconds':init,'train_seconds':train,'add_seconds':add,
                'search_seconds_samples':samples,'search_seconds_median':float(np.median(samples)),
                'qps':len(xq)/float(np.median(samples)),'recall':float(recall.mean()),
                'recall_by_query':recall.tolist(),'query_transform_seconds_samples':transforms,**memory})
        helpers.write_results(args.output,report)
        print(f'[{number}/{len(cases)}] {family}{bits}/{mode} seed={seed}: 14 nprobe/k rows saved',flush=True)
    report['status']='complete'
    helpers.write_results(args.output,report)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--npz',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'amumu-evidence/results/msong_ann_native.json')
    parser.add_argument('--pad-baselines',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--threads',type=int,default=1,choices=(1,))
    args=parser.parse_args()
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        assert all(pool['num_threads']==1 for pool in threadpool_info())
        run(args)

