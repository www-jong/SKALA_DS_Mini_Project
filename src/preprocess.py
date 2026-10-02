"""원본 MAT를 읽어 프로그램에서 재사용할 중간 데이터를 준비한다.

흐름: data/raw → _eda 스크립트 3개 → .cache/preprocessing.
원본 파일의 크기·수정 시각이 바뀌거나 필수 중간 파일이 없으면 다시 추출한다.
여기서는 대치·표준화·모델 학습을 하지 않는다."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pickle
import hashlib
import numpy as np
import pandas as pd
PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / '.cache/preprocessing'


def source_fingerprint(dataset):
    """원본 3개 파일의 이름·크기·수정 시각을 기록해 반환한다. 파일 내용을 해시하는 함수는 아니다."""
    dataset = Path(dataset).expanduser().resolve()
    names = json.loads((PROJECT / 'config.json').read_text())['cohort']['mat_files']
    sources = []
    for batch, name in names.items():
        path = dataset / name
        if not path.is_file():
            raise FileNotFoundError(f'Prepare original MAT in data/raw: {name}')
        stat = path.stat()
        sources.append({'batch': batch, 'file': name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    # 원본뿐 아니라 추출 규칙이 바뀐 경우에도 이전 표를 그대로 쓰지 않는다.
    scripts = [PROJECT / 'src/preprocess.py', *sorted((PROJECT / 'src/_eda').glob('*.py'))]
    return {
        'pipeline_version': 2,
        'sources': sources,
        'cohort': json.loads((PROJECT / 'config.json').read_text())['cohort'],
        'extractor_sha256': {
            str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in scripts
        },
    }


def ensure_preprocessed():
    """필수 중간 파일과 원본 기록을 확인하고, 재추출이 필요할 때만 rebuild를 호출한다."""
    dataset = Path(os.environ.get('BATTERY_DATA_DIR', str(PROJECT / 'data/raw')))
    expected = source_fingerprint(dataset)
    manifest = PROCESSED / 'source_manifest.json'
    # 기본 표와 셀 배열이 모두 있어야 다음 단계가 독립적으로 실행된다.
    required = ['features.csv', 'cell_audit.csv', 'cell_arrays.npz']
    try:
        current = json.loads(manifest.read_text()) if manifest.exists() else None
    except json.JSONDecodeError:
        current = None
    if current != expected or not all(((PROCESSED / name).is_file() for name in required)):
        rebuild(dataset)


def load_cells():
    """숫자 NPZ를 {셀 ID: summary·Q(V)·전류 배열} 형태의 dict로 복원해 반환한다."""
    ensure_preprocessed()
    with np.load(PROCESSED / 'cell_arrays.npz', allow_pickle=False) as z:
        cells = {}
        for i, cell in enumerate(z['cell']):
            # 셀마다 관측 사이클 수가 달라서, 이어 붙인 배열의 시작·끝 위치로 잘라 복원한다.
            a, b = z['summary_offsets'][i:i + 2]
            cells[str(cell)] = {
                'summary': {k.removeprefix('summary_'): z[k][a:b].copy() for k in z.files if k.startswith('summary_') and k != 'summary_offsets'},
                **{k: z[k][i].copy() for k in ['v', 'q10', 'q100', 'dq']},
            }
            if 'current_offsets' in z:
                ca, cb = z['current_offsets'][i:i + 2]
                cells[str(cell)]['current'] = {k: z['current_' + k][ca:cb].copy() for k in ['t', 'I', 'Qc']}
    return cells


def tables():
    """채택 셀의 기본 피처 표와 전체 셀의 채택·제외 기록을 순서대로 반환한다."""
    ensure_preprocessed()
    return (pd.read_csv(PROCESSED / 'features.csv'), pd.read_csv(PROCESSED / 'cell_audit.csv'))


def rebuild(dataset):
    """MAT 디렉토리에서 EDA 스크립트를 실행하고, 셀 배열을 숫자 NPZ로 직렬화한다."""
    dataset = Path(dataset).expanduser().resolve()
    config = json.loads((PROJECT / 'config.json').read_text())
    for name in config['cohort']['mat_files'].values():
        if not (dataset / name).exists():
            raise FileNotFoundError(dataset / name)
    env = dict(os.environ, BATTERY_DATA_DIR=str(dataset))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    # 기본 추출이 먼저다. 뒤의 두 스크립트는 그 결과로 추가 통계와 그림을 만든다.
    for script in ['extract.py', 'supplement.py', 'expand.py']:
        subprocess.run([sys.executable, str(PROJECT / 'src/_eda' / script)], env=env, check=True)
    # 여기서 읽는 pickle은 바로 위 추출 과정에서 직접 만든 로컬 중간 파일이다.
    with open(PROCESSED / 'extracted.pkl', 'rb') as f:
        cache = pickle.load(f)
    ids = pd.read_csv(PROCESSED / 'features.csv').cell.tolist()
    arr = {'cell': np.array(ids, dtype=str)}
    lengths = [len(cache[k]['summary']['cycle']) for k in ids]
    # 가변 길이 summary는 하나로 이어 붙이고, 셀별 경계 위치를 따로 저장한다.
    arr['summary_offsets'] = np.r_[0, np.cumsum(lengths)]
    for key in cache[ids[0]]['summary']:
        arr['summary_' + key] = np.concatenate([cache[k]['summary'][key] for k in ids])
    # Q(V)는 모든 셀이 1,000개 전압 지점을 사용하므로 셀 × 전압의 행렬로 저장한다.
    for key in ['v', 'q10', 'q100', 'dq']:
        arr[key] = np.stack([cache[k][key] for k in ids])
    # 전류 기록도 셀마다 길이가 달라 summary와 같은 방식으로 보관한다.
    lengths = [len(cache[k]['current']['t']) for k in ids]
    arr['current_offsets'] = np.r_[0, np.cumsum(lengths)]
    for key in ['t', 'I', 'Qc']:
        arr['current_' + key] = np.concatenate([cache[k]['current'][key] for k in ids])
    np.savez_compressed(PROCESSED / 'cell_arrays.npz', **arr)
    # 모든 작업이 끝난 뒤 원본 기록을 남긴다. 중간 실패를 완료로 취급하지 않기 위해서다.
    (PROCESSED / 'source_manifest.json').write_text(
        json.dumps(source_fingerprint(dataset), indent=2) + '\n',
    )
    print('Rebuilt audit, EDA tables and numeric arrays.')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=os.environ.get('BATTERY_DATA_DIR', str(PROJECT / 'data/raw')))
    p.add_argument('--rebuild', action='store_true')
    args = p.parse_args()
    if args.rebuild:
        rebuild(args.dataset)
    else:
        # --dataset만 지정한 경우에도 그 경로를 사용해 준비 여부를 확인한다.
        os.environ['BATTERY_DATA_DIR'] = args.dataset
        print(tables()[0].groupby('batch').size())
