"""셀 하나를 피처 테이블의 한 행으로 만든다.

입력은 preprocess가 준비한 셀별 배열과 초기 기본 피처다.
새 피처는 cycle 2–100 또는 Q100(V)−Q10(V)에서만 계산한다.
73개는 탐색 후보의 전체 개수이며, 모델에는 선택한 피처군만 넣는다.
중앙값 대치나 표준화는 train.py의 학습 fold 안에서 처리한다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
from src.preprocess import load_cells, ensure_preprocessed
OUT = ROOT / '.cache' / 'features'
OUT.mkdir(parents=True, exist_ok=True)


def slope(x, y):
    """유한한 x, y 관측 3개 이상으로 직선 기울기를 구한다. 부족하면 NaN을 반환한다."""
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[m], y[m], 1)[0]) if m.sum() >= 3 else np.nan


def build_features():
    """초기 후보 73개를 생성·저장하고 (피처 DataFrame, 피처 정의 dict)를 반환한다."""
    ensure_preprocessed()
    config = json.loads((ROOT / 'config.json').read_text())
    base_table = pd.read_csv(ROOT / '.cache/preprocessing/features.csv')
    # EDA 표에는 전체 수명 분석 값도 있다. 설정에 명시한 초기 피처만 골라 가져온다.
    base = list(
        dict.fromkeys(
            config['features']['core'] + sum(
                [v for k, v in config['features']['alternatives_from_features_csv'].items() if k != 'tag_sensitivity'],
                [],
            ),
        ),
    )
    assert not set(base) & set(config['features']['forbidden_inputs'])
    cell_arrays = load_cells()
    rows = []
    definitions = {}
    for cell_record in base_table.itertuples():
        cell_data = cell_arrays[cell_record.cell]
        summary = cell_data['summary']
        x = np.asarray(summary['cycle'])
        # 예측 시점은 100사이클이다. 이후 관측이 새 피처 계산에 들어가지 않도록 먼저 자른다.
        early = (x >= 2) & (x <= 100)
        xx = x[early]
        q = np.asarray(summary['QDischarge'])[early].copy()
        q[(q <= 0) | (q > 1.3)] = np.nan
        row = {k: getattr(cell_record, k) for k in base}
        # 셀 ID·배치·수명은 분리와 평가용 정보다. 아래 피처군 목록에서는 제외한다.
        row.update(
            cell=cell_record.cell,
            batch=cell_record.batch,
            policy=cell_record.policy,
            policy_group=f'{cell_record.c1:g}|{cell_record.soc_switch:g}|{cell_record.c2:g}',
            life=cell_record.life,
            target_log10=np.log10(cell_record.life),
            policy_newstructure=float(cell_record.newstructure),
        )

        def add(name, value, definition):
            """한 셀의 피처 값과 공통 피처 정의를 함께 기록한다."""
            row[name] = float(value)
            definitions[name] = definition

        def at(a, n):
            """잘라 둔 초기 사이클 배열에서 지정 사이클의 값을 찾는다. 위치를 사이클 번호로 가정하지 않는다."""
            v = a[xx == n]
            return float(v[0]) if len(v) else np.nan
        # 용량의 크기와 상대 변화: Q는 Ah, 상대 용량은 Q2로 나눈 무차원 값이다.
        for n in [10, 20, 50, 80]:
            val = at(q, n)
            add(f'q_at_{n}', val, f'QDischarge cycle{n}, Ah')
            add(f'q_relative_{n}', val / cell_record.q2, f'QDischarge({n})/QDischarge(2)')
        add('q_growth_2_100', cell_record.q100 / cell_record.q2 - 1, 'QDischarge100/QDischarge2 -1')
        add('q_early_std', np.nanstd(q, ddof=1), 'std QDischarge cycles2..100, Ah')
        # 같은 초기 구간도 앞·중간·뒤의 기울기가 다를 수 있어 세 구간을 따로 본다.
        slopes = []
        for a, b in [(2, 30), (31, 60), (61, 100)]:
            m = (xx >= a) & (xx <= b)
            z = slope(xx[m], q[m])
            slopes.append(z)
            add(f'q_slope_{a}_{b}', z, f'OLS QDischarge slope cycles{a}..{b}, Ah/cycle')
        add('q_slope_acceleration', slopes[-1] - slopes[0], 'early-window slope61..100 minus slope2..30')
        # 저항·온도·충전 시간의 변동과 추세를 계산한다. 0 이하의 센서 값은 결측으로 둔다.
        for source, prefix in [('IR', 'ir'), ('Tmax', 'tmax'), ('Tavg', 'tavg'), ('chargetime', 'charge_minutes')]:
            a = np.asarray(summary[source])[early].copy()
            a[(a <= 0) | ~np.isfinite(a)] = np.nan
            add(
                prefix + '_early_std',
                np.nanstd(a, ddof=1) if np.isfinite(a).sum() > 1 else np.nan,
                f'std {source} cycles2..100',
            )
            add(prefix + '_early_slope', slope(xx, a), f'OLS {source} slope cycles2..100')
            v = a[xx <= 10]
            first = np.nanmean(v) if np.isfinite(v).any() else np.nan
            v = a[xx >= 91]
            last = np.nanmean(v) if np.isfinite(v).any() else np.nan
            add(prefix + '_window_change', last - first, f'mean {source} cycles91..100 - mean cycles2..10')
        # 충전량 대비 방전량 비율도 후보로 만든다. 결측 대치는 여기서 하지 않는다.
        qc = np.asarray(summary['QCharge'])[early].copy()
        qc[(qc <= 0) | (qc > 1.3)] = np.nan
        eff = q / qc
        add('coulombic_eff_mean', np.nanmean(eff), 'mean QDischarge/QCharge cycles2..100')
        add('coulombic_eff_slope', slope(xx, eff), 'OLS QDischarge/QCharge cycles2..100')
        # ΔQ(V) = Q100(V) − Q10(V). 용량 곡선의 변화 폭과 전압별 형태를 함께 살펴본다.
        dq = np.asarray(cell_data['dq'])
        v = np.asarray(cell_data['v'])
        assert len(dq) == 1000 and np.isfinite(dq).all()
        add('dq_iqr', np.quantile(dq, 0.75) - np.quantile(dq, 0.25), 'IQR Q100(V)-Q10(V), Ah')
        add('dq_abs_mean', np.mean(np.abs(dq)), 'mean absolute Q100(V)-Q10(V), Ah')
        add('dq_range', np.ptp(dq), 'range Q100(V)-Q10(V), Ah')
        # 원본 전압 축은 내림차순이다. np.interp에 맞춰 전압과 ΔQ를 함께 뒤집는다.
        for voltage in [2.2, 2.5, 2.8, 3.0, 3.2, 3.4]:
            add(
                'dq_at_' + str(voltage).replace('.', 'p'),
                np.interp(voltage, v[::-1], dq[::-1]),
                f'Q100-Q10 interpolated at {voltage}V, Ah',
            )
        # 전체 분산뿐 아니라 어느 전압 구간에서 변화가 커지는지도 후보 피처로 남긴다.
        for a, b in [(2, 2.5), (2.5, 3), (3, 3.3), (3.3, 3.5)]:
            z = dq[(v >= a) & (v <= b)]
            tag = f'{a:g}_{b:g}'.replace('.', 'p')
            add(
                'dq_band_logvar_' + tag,
                np.log10(max(np.var(z, ddof=1), 1e-20)),
                f'log10 sample variance of deltaQ in {a}..{b}V',
            )
            add('dq_band_mean_' + tag, np.mean(z), f'mean deltaQ in {a}..{b}V, Ah')
        # 논문 피처 정의에 맞춘 통계다. 왜도는 논문 수식의 길이 보정, 첨도는 Pearson 정의를 쓴다.
        from scipy.stats import skew, kurtosis
        add('dq_logmin', np.log10(abs(np.min(dq))), 'log10 absolute minimum deltaQ, paper')
        add(
            'dq_logskew',
            np.log10(abs(skew(dq, bias=True))) - 1.5 * np.log10(len(dq)),
            'printed paper skewness; standard log-skew minus4.5 for1000 points',
        )
        add(
            'dq_logkurtosis',
            np.log10(abs(kurtosis(dq, fisher=False, bias=True))),
            'log10 Pearson kurtosis deltaQ',
        )
        add('q_max_minus_q2', np.nanmax(q) - cell_record.q2, 'max QDischarge cycles2..100 minus Q2, Ah')
        rows.append(row)
    feature_table = pd.DataFrame(rows)
    protocol = ['c1', 'c2', 'soc_switch', 'c_eff', 'policy_newstructure']
    # 타깃과 식별 정보를 빼고 모델에 넣을 수 있는 숫자 후보만 모은다.
    all_features = [k for k in feature_table if k not in ['cell', 'batch', 'policy', 'policy_group', 'life', 'target_log10']]
    shape = [k for k in all_features if k.startswith('dq_')]
    # 73개 후보를 전부 쓰는 것은 아니다. 비교 목적에 따라 작은 피처군부터 묶어 둔다.
    families = {
        'dq_only': ['dq_logvar'],
        'dq_shape': shape,
        'core6': config['features']['core'],
        'engineered_no_policy': [k for k in all_features if k not in protocol],
        'engineered_with_policy': all_features,
    }
    families['paper_discharge6'] = ['dq_logmin', 'dq_logvar', 'dq_logskew', 'dq_logkurtosis', 'q2', 'q_max_minus_q2']
    families['capacity'] = ['dq_logvar', 'q2', 'q_slope_last']
    families['capacity_ir_temperature'] = ['dq_logvar', 'q2', 'q_slope_last', 'ir_delta', 'tmax_mean']
    assert len(feature_table) == 118 and feature_table.cell.is_unique
    assert not set(all_features) & set(config['features']['forbidden_inputs'])
    assert not np.isinf(feature_table[all_features].to_numpy()).any()
    # 셀별 계산 결과만 저장한다. 대치·표준화는 train.py에서 각 학습 fold에 맞춰 적합한다.
    feature_table.to_csv(OUT / 'engineered_features.csv', index=False)
    raw = np.stack([cell_arrays[k]['dq'] for k in feature_table.cell])
    np.savez_compressed(OUT / 'deltaq_curves.npz', cell=feature_table.cell.to_numpy(dtype=str), dq=raw)
    catalog = {
        'prediction_cycle': 100,
        'n_cells': 118,
        'n_numeric_features': len(all_features),
        'base_features': base,
        'new_definitions': definitions,
        'feature_families': families,
        'missing': feature_table[all_features].isna().sum().to_dict(),
        'rule': 'All new features are per-cell cycle<=100 or Q100-Q10; global imputation/scaling/PCA/selection must be train-fold fit',
        'policy_sensitivity': 'engineered_with_policy includes numeric protocol and newstructure tag; exclude batch, cell IDs and whole-life metrics',
    }
    (OUT / 'feature_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    print(
        f'Engineered {len(all_features)} numeric features for {len(feature_table)} cells; families=' + str({k: len(v) for k, v in families.items()}),
    )
    return (feature_table, catalog)

if __name__ == '__main__':
    build_features()
