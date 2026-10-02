"""MATLAB v7.3 원본의 셀 참조를 읽어 셀별 요약·곡선과 EDA 표를 만든다.

preprocess.rebuild()가 첫 번째로 실행하는 내부 스크립트다.
모든 사이클을 읽는 이유는 EDA의 열화 곡선·Knee 분석 때문이다.
전체 수명 지표도 저장되므로 모델 입력에는 반드시 whitelist를 사용한다."""
from pathlib import Path
import os
import json
import re
import pickle
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / '.cache/preprocessing'
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr, pearsonr, skew, kurtosis, kruskal
from scipy.ndimage import median_filter
DATA = Path(os.environ.get('BATTERY_DATA_DIR', str(PROJECT / 'data/raw')))
ASSETS = PROJECT / 'results/figures'
ASSETS.mkdir(parents=True, exist_ok=True)
_font = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
if _font.exists():
    font_manager.fontManager.addfont(str(_font))
plt.rcParams.update(
    {
        'font.family': ['DejaVu Sans', 'Apple SD Gothic Neo'],
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.titleweight': 'bold',
        'axes.grid': True,
        'grid.alpha': 0.17,
    },
)
COLORS = {'B1': '#167c80', 'B2': '#dd7745', 'B3': '#6162aa'}
FILES = {
    'B1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'B2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'B3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}
LABELS = {
    'dq_logvar': 'log₁₀ Var(ΔQ)',
    'dq_logabsmin': 'log₁₀ |min ΔQ|',
    'dq_mean': '평균 ΔQ',
    'dq_skew': 'ΔQ 왜도',
    'dq_kurtosis': 'ΔQ 첨도',
    'dq_logstd': 'log₁₀ Std(ΔQ)',
    'q2': 'Qd(2)',
    'q100': 'Qd(100)',
    'q_ratio': 'Qd(100)/Qd(2)',
    'q_slope': 'Qd 기울기 (10–100)',
    'q_slope_last': 'Qd 기울기 (91–100)',
    'ir_mean': '초기 평균 IR',
    'ir_delta': 'IR(100)−IR(10)',
    'tmax_mean': '초기 평균 Tmax',
    'tavg_mean': '초기 평균 Tavg',
    'temp_span': '초기 온도 범위',
    'charge_time': '초기 충전 시간',
    'c1': '1단계 C-rate',
    'c2': '2단계 C-rate',
    'soc_switch': '전환 SOC',
    'c_eff': '프로토콜 유효 C-rate',
    'i_mean': '실측 평균 C-rate',
    'i_std': '실측 전류 변동',
    'i_high': '실측 >5C 시간 비율',
    'i_peak': '실측 최대 C-rate',
}
FEATURES = list(LABELS)


def scalar(f, r):
    """MATLAB 참조가 가리키는 단일 숫자를 Python float로 읽는다."""
    return float(np.asarray(f[r]).item())


def curve(f, g, k, j):
    """특정 사이클 j의 필드 배열을 참조에서 읽어 1차원으로 반환한다. j는 0부터 시작한다."""
    return np.asarray(f[g[k][j, 0]], dtype=float).ravel()


def fitline(x, y):
    """유효한 사이클·용량 쌍으로 직선 기울기를 계산한다."""
    m = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[m], y[m], 1)[0] if m.sum() >= 3 else np.nan


def knee_fit(x, y, size=21):
    """평활화한 Qd에 연속 두 구간 회귀를 적합해 Knee와 전후 기울기를 추정한다."""
    m = np.isfinite(x) & np.isfinite(y) & (x >= 20) & (y > 0.5) & (y < 1.3)
    x = x[m]
    y = y[m]
    if len(x) < 100:
        return {
            'knee': np.nan,
            'early_slope': np.nan,
            'late_slope': np.nan,
            'accel_ratio': np.nan,
            'knee_gain': np.nan,
        }
    y = median_filter(y, size=size, mode='nearest')
    ids = np.linspace(0, len(x) - 1, min(len(x), 350)).astype(int)
    xx = x[ids]
    yy = y[ids]
    base = np.column_stack([np.ones(len(xx)), xx])
    b = np.linalg.lstsq(base, yy, rcond=None)[0]
    sse0 = np.sum((yy - base @ b) ** 2)
    best = None
    for k in np.linspace(np.quantile(xx, 0.2), np.quantile(xx, 0.85), 100):
        a = np.column_stack([base, np.maximum(0, xx - k)])
        coef = np.linalg.lstsq(a, yy, rcond=None)[0]
        sse = np.sum((yy - a @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, k, coef)
    sse, k, coef = best
    early = coef[1]
    late = early + coef[2]
    ratio = abs(late) / max(abs(early), 1e-08)
    gain = 1 - sse / max(sse0, 1e-15)
    accepted = early < 0 and late < early and (ratio >= 2) and (gain >= 0.25)
    return {
        'knee': k if accepted else np.nan,
        'early_slope': early,
        'late_slope': late,
        'accel_ratio': ratio,
        'knee_gain': gain,
    }


def current_features(f, c, j):
    """한 사이클의 충전 구간에서 시간 가중 평균·표준편차·5C 초과 비율·최대값·시간을 반환한다."""
    t = curve(f, c, 't', j)
    i = curve(f, c, 'I', j)
    qc = curve(f, c, 'Qc', j)
    # 기록 간격이 일정하지 않아 샘플 개수가 아니라 경과 시간으로 가중한다. I는 데이터의 C-rate 값이다.
    dt = np.diff(t)
    im = (i[:-1] + i[1:]) / 2
    qm = (qc[:-1] + qc[1:]) / 2
    m = np.isfinite(dt) & np.isfinite(im) & np.isfinite(qm) & (dt > 0) & (im > 0.05) & (qm <= 0.88)
    if not m.any():
        return [np.nan] * 5
    w = dt[m]
    rates = im[m]
    mu = np.average(rates, weights=w)
    return [
        mu,
        np.sqrt(np.average((rates - mu) ** 2, weights=w)),
        np.average(rates > 5, weights=w),
        np.max(rates),
        w.sum(),
    ]
rows = []
cells = {}
audit_extra = []
# MAT v7.3의 summary·cycles 필드는 HDF5 객체 참조다. 참조를 따라가 셀별 배열을 읽는다.
for batch, name in FILES.items():
    print('Extracting', batch, flush=True)
    with h5py.File(DATA / name, 'r') as f:
        b = f['batch']
        for idx in range(len(b['summary'])):
            ident = f'{batch}-C{idx + 1:02d}'
            s = f[b['summary'][idx, 0]]
            c = f[b['cycles'][idx, 0]]
            summ = {k: np.asarray(s[k], dtype=float).ravel() for k in s}
            x = summ['cycle']
            q = summ['QDischarge']
            life = scalar(f, b['cycle_life'][idx, 0])
            policy = ''.join((chr(int(v)) for v in np.asarray(f[b['policy_readable'][idx, 0]]).ravel()))
            v = np.asarray(f[b['Vdlin'][idx, 0]], dtype=float).ravel()
            # 수명 라벨이 없거나 EOL 도달이 확인되지 않은 셀, 공식 수집 문제 셀을 제외한다.
            reason = ''
            if not np.isfinite(life):
                reason = '수명 라벨 없음 (VarCharge/SLOWCYCLE)' if batch == 'B2' else '우측 검열: EOL 미관측'
            elif q[-1] > 0.885:
                reason = '우측 검열: 종료 Qd > 0.885 Ah'
            elif batch == 'B3' and idx == 37:
                reason = '공식 로더에서 수집 문제 지정 (MATLAB 38번)'
            row = {
                'cell': ident,
                'batch': batch,
                'file': name,
                'index_1based': idx + 1,
                'policy': policy,
                'life_raw': life,
                'life': life if not reason else np.nan,
                'included': not bool(reason),
                'exclusion': reason,
                'n_cycles': len(x),
                'q_end': q[-1],
                'q_invalid_count': int(np.sum((q <= 0) | (q > 1.3) | ~np.isfinite(q))),
                'n_cycle_records': c['Qdlin'].shape[0],
                'summary_cycle_contiguous': bool(np.array_equal(x, np.arange(1, len(x) + 1))),
            }
            hits = x[(q > 0) & (q < 0.88)]
            row['first_below_088'] = hits[0] if len(hits) else np.nan

            def at(key, n):
                """summary의 실제 사이클 번호로 값을 찾으며 양수가 아니면 결측으로 취급한다."""
                a = summ[key]
                positions = np.where(x == n)[0]
                return a[positions[0]] if len(positions) and np.isfinite(a[positions[0]]) and (a[positions[0]] > 0) else np.nan
            # 모델 입력에 쓸 기본 통계는 2–100사이클만 사용한다. 전체 수명 통계는 EDA 설명용이다.
            early = (x >= 2) & (x <= 100)
            for key, out in [
                ('IR', 'ir_mean'),
                ('Tmax', 'tmax_mean'),
                ('Tavg', 'tavg_mean'),
                ('chargetime', 'charge_time'),
            ]:
                a = summ[key][early]
                a = a[(a > 0) & np.isfinite(a)]
                row[out] = float(np.mean(a)) if len(a) else np.nan
            row.update(
                q2=at('QDischarge', 2),
                q100=at('QDischarge', 100),
                ir_delta=at('IR', 100) - at('IR', 10),
            )
            row['q_ratio'] = row['q100'] / row['q2']
            row['q_slope'] = fitline(
                x[(x >= 10) & (x <= 100)],
                np.where(
                    (q[(x >= 10) & (x <= 100)] > 0) & (q[(x >= 10) & (x <= 100)] < 1.3),
                    q[(x >= 10) & (x <= 100)],
                    np.nan,
                ),
            )
            row['q_slope_last'] = fitline(x[(x >= 91) & (x <= 100)], q[(x >= 91) & (x <= 100)])
            temp = (summ['Tmax'] - summ['Tmin'])[early]
            row['temp_span'] = float(np.nanmean(temp))
            match = re.search('([\\d.]+)C\\((\\d+)%\\)-([\\d.]+)C', policy)
            if match:
                c1, soc, c2 = map(float, match.groups())
                soc /= 100
                row.update(
                    c1=c1,
                    c2=c2,
                    soc_switch=soc,
                    c_eff=0.8 / (soc / c1 + (0.8 - soc) / c2),
                    step_delta=c2 - c1,
                )
            else:
                row.update(c1=np.nan, c2=np.nan, soc_switch=np.nan, c_eff=np.nan, step_delta=np.nan)
            row['newstructure'] = 'newstructure' in policy
            if len(x) >= 100:
                assert row['summary_cycle_contiguous']
                q10 = curve(f, c, 'Qdlin', 9)
                q100 = curve(f, c, 'Qdlin', 99)
                dq = q100 - q10
                row['dq_valid'] = bool(
                    len(dq) == len(v) == 1000 and np.isfinite(dq).all() and (np.max(np.abs(q10)) < 1.5) and (np.max(np.abs(q100)) < 1.5),
                )
                if row['dq_valid']:
                    var = float(np.var(dq, ddof=1))
                    row.update(
                        dq_logvar=np.log10(max(var, 1e-20)),
                        dq_logstd=np.log10(max(np.std(dq, ddof=1), 1e-20)),
                        dq_logabsmin=np.log10(max(abs(np.min(dq)), 1e-20)),
                        dq_mean=np.mean(dq),
                        dq_skew=skew(dq, bias=False),
                        dq_kurtosis=kurtosis(dq, bias=False),
                    )
                currents = np.array([current_features(f, c, j) for j in [9, 99]])
                for j, k in enumerate(['i_mean', 'i_std', 'i_high', 'i_peak', 'i_duration']):
                    row[k] = float(np.nanmean(currents[:, j]))
                cells[ident] = {
                    'summary': summ,
                    'v': v,
                    'q10': q10,
                    'q100': q100,
                    'dq': dq,
                    'current': {'t': curve(f, c, 't', 9), 'I': curve(f, c, 'I', 9), 'Qc': curve(f, c, 'Qc', 9)},
                }
            end = life if row['included'] else x[-1]
            m = (x <= end) & (q > 0) & (q < 1.3)
            result = knee_fit(x[m], q[m])
            row.update(result)
            if np.isfinite(row['knee']):
                ks = [knee_fit(x[m], q[m], z)['knee'] for z in [11, 21, 41]]
                ks = np.array(ks)
                row['knee_smoothing_span'] = np.nanmax(ks) - np.nanmin(ks)
                row['knee_fraction'] = row['knee'] / end
            rows.append(row)
df = pd.DataFrame(rows)
d = df[df.included].copy()
df.to_csv(ROOT / 'cell_audit.csv', index=False)
d.to_csv(ROOT / 'features.csv', index=False)
with open(ROOT / 'extracted.pkl', 'wb') as f:
    pickle.dump(cells, f)
summary = []
for b, g in d.groupby('batch'):
    raw = df[df.batch == b]
    short = int((g.life < 500).sum())
    long = int((g.life > 1000).sum())
    q1, q3 = g.life.quantile([0.25, 0.75])
    low = q1 - 1.5 * (q3 - q1)
    high = q3 + 1.5 * (q3 - q1)
    summary.append(
        {
            'batch': b,
            'raw': len(raw),
            'usable': len(g),
            'excluded': len(raw) - len(g),
            'mean': g.life.mean(),
            'median': g.life.median(),
            'std': g.life.std(),
            'min': g.life.min(),
            'max': g.life.max(),
            'q25': q1,
            'q75': q3,
            'short': short,
            'long': long,
            'short_pct': short / len(g) * 100,
            'long_pct': long / len(g) * 100,
            'outlier_low_fence': low,
            'outlier_high_fence': high,
            'lower_outliers': g.loc[g.life < low, 'cell'].tolist(),
            'upper_outliers': g.loc[g.life > high, 'cell'].tolist(),
            'knee_n': int(g.knee.notna().sum()),
            'knee_frac_med': g.knee_fraction.median(),
            'accel_med': g.accel_ratio.median(),
            'q_gain_pct': (g.q_ratio > 1).mean() * 100,
        },
    )
bs = pd.DataFrame(summary)
bs.to_csv(ROOT / 'batch_summary.csv', index=False)
rng = np.random.default_rng(20261001)


def corr(x, y):
    """유한한 셀 단위 값으로 Spearman 상관·p값·셀 수를 반환한다."""
    m = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[m]
    y = np.asarray(y)[m]
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan, len(x))
    r, p = spearmanr(x, y)
    return (float(r), float(p), len(x))
corrows = []
for batch, g in [('ALL', d), *list(d.groupby('batch'))]:
    for feature in FEATURES:
        rho, p, n = corr(g[feature].to_numpy(), g.life.to_numpy())
        m = g[[feature, 'life']].dropna()
        r = pearsonr(m[feature], np.log10(m.life)).statistic if len(m) > 3 and m[feature].std() > 0 else np.nan
        corrows.append(
            {
                'batch': batch,
                'feature': feature,
                'spearman': rho,
                'p_exploratory': p,
                'n': n,
                'pearson_loglife': r,
            },
        )
co = pd.DataFrame(corrows)
co.to_csv(ROOT / 'correlations.csv', index=False)
# 전체 상관에 배치 차이가 섞였는지 확인하려고 순위에서 배치별 평균을 빼고 다시 비교한다.
partial = []
for feat in FEATURES:
    g = d[[feat, 'life', 'batch']].dropna().copy()
    g['a'] = g[feat].rank()
    g['b'] = g.life.rank()
    groups = pd.get_dummies(g.batch, dtype=float)
    a = g.a.to_numpy().copy()
    b = g.b.to_numpy().copy()
    a -= groups.to_numpy() @ np.linalg.lstsq(groups, a, rcond=None)[0]
    b -= groups.to_numpy() @ np.linalg.lstsq(groups, b, rcond=None)[0]
    partial.append(
        {
            'feature': feat,
            'batch_adjusted_rank_r': float(pearsonr(a, b).statistic) if np.std(a) > 1e-09 else np.nan,
        },
    )
pa = pd.DataFrame(partial)
pa.to_csv(ROOT / 'batch_adjusted_correlations.csv', index=False)
# 부트스트랩으로 상관의 불확실성을 살펴본다. 위의 고정 난수 시드로 재실행 결과를 맞춘다.
ci = {}
for b, g in [('ALL', d), *list(d.groupby('batch'))]:
    a = g[['dq_logvar', 'life']].dropna().to_numpy()
    boot = []
    for _ in range(2000):
        z = a[rng.integers(0, len(a), len(a))]
        boot.append(spearmanr(z[:, 0], z[:, 1]).statistic)
    ci[b] = [float(v) for v in np.nanquantile(boot, [0.025, 0.975])]
fc = d[FEATURES].corr(method='spearman')
pairs = []
for i, a in enumerate(FEATURES):
    for b in FEATURES[i + 1:]:
        if abs(fc.loc[a, b]) >= 0.9:
            pairs.append({'feature1': a, 'feature2': b, 'rho': fc.loc[a, b]})
pd.DataFrame(pairs).to_csv(ROOT / 'collinearity_pairs.csv', index=False)
shortlist = [
    'dq_logvar',
    'dq_logabsmin',
    'dq_mean',
    'q2',
    'q_ratio',
    'q_slope',
    'ir_mean',
    'ir_delta',
    'tmax_mean',
    'charge_time',
    'c_eff',
    'i_mean',
]
vifrows = []
# VIF는 결측 없는 셀에서 계산하는 EDA 진단이다. 모델 학습의 결측 처리와는 별개다.
z = d[shortlist].dropna()
z = (z - z.mean()) / z.std()
arr = z.to_numpy()
for j, k in enumerate(shortlist):
    others = np.delete(arr, j, axis=1)
    pred = others @ np.linalg.lstsq(others, arr[:, j], rcond=None)[0]
    r2 = 1 - np.sum((arr[:, j] - pred) ** 2) / np.sum(arr[:, j] ** 2)
    vifrows.append({'feature': k, 'vif': 1 / max(1 - r2, 1e-12), 'n': len(z)})
pd.DataFrame(vifrows).to_csv(ROOT / 'vif.csv', index=False)
prot = d.groupby(['batch', 'policy']).life.agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index(
)
prot.to_csv(ROOT / 'protocol_summary.csv', index=False)
q4 = []
for b, g in [('ALL', d), *list(d.groupby('batch'))]:
    for feat in ['c1', 'c2', 'c_eff', 'i_mean', 'i_std', 'i_high', 'charge_time', 'step_delta']:
        for target in ['life', 'late_slope', 'q_slope']:
            rho, p, n = corr(g[feat].to_numpy(), g[target].to_numpy())
            q4.append({'batch': b, 'feature': feat, 'target': target, 'rho': rho, 'n': n, 'p': p})
pd.DataFrame(q4).to_csv(ROOT / 'charge_correlations.csv', index=False)


def save(name):
    """현재 그림을 제출용 figures에 저장하고 닫는다."""
    plt.savefig(ASSETS / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True, sharey=True)
# Q1: 수명 분포와 장·단수명 비율.
bins = np.arange(150, 2351, 100)
for ax, (b, g) in zip(axes, d.groupby('batch')):
    ax.hist(g.life, bins=bins, color=COLORS[b], edgecolor='white')
    ax.axvline(500, c='#dd7745', ls='--')
    ax.axvline(1000, c='#167c80', ls='--')
    ax.set(
        title=f'{b} · n={len(g)} · 중앙값 {g.life.median():.0f}',
        xlabel='Cycle Life (cycles)',
        xlim=(150, 2300),
    )
    ax.set_xticks([150, 500, 1000, 1500, 2000, 2300])
    ax.tick_params(axis='x', labelsize=9)
axes[0].set_ylabel('셀 수')
fig.suptitle('Q1 | 150–2,300 cycles: 배치별 수명 분포', y=1.04)
save('q1_histogram')
fig, axes = plt.subplots(1, 2, figsize=(12, 3.3))
positions = np.arange(3)
axes[0].boxplot([d[d.batch == b].life for b in COLORS], tick_labels=list(COLORS), showmeans=True)
axes[0].set_ylabel('Cycle Life (cycles)')
for b, g in d.groupby('batch'):
    for _, r in g.iterrows():
        axes[0].scatter(
            list(COLORS).index(b) + 1 + rng.uniform(-0.14, 0.14),
            r.life,
            s=12,
            color=COLORS[b],
            alpha=0.7,
        )
bottom = np.zeros(3)
for mask, col, label in [
    (lambda x: x < 500, '#dd7745', '단수명 <500'),
    (lambda x: (x >= 500) & (x <= 1000), '#b5bdc8', '중간 500–1,000'),
    (lambda x: x > 1000, '#167c80', '장수명 >1,000'),
]:
    vals = np.array([mask(d[d.batch == b].life).mean() * 100 for b in COLORS])
    axes[1].bar(list(COLORS), vals, bottom=bottom, color=col, label=label)
    for j, val in enumerate(vals):
        if val > 0:
            axes[1].text(
                j,
                bottom[j] + val / 2,
                f'{val:.1f}%',
                ha='center',
                va='center',
                color='white' if col != '#b5bdc8' else '#233',
            )
    bottom += vals
axes[1].set_ylabel('확정 라벨 셀 내 비율 (%)')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=9)
save('q1_groups')
# Q2: 전체 열화와 초기 100사이클을 나누어 본다. Knee 등 전체 수명 값은 모델 입력이 아니다.
for mode in ['full', 'early']:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, (b, g) in zip(axes, d.groupby('batch')):
        for _, r in g.iterrows():
            s = cells[r.cell]['summary']
            x = s['cycle']
            q = s['QDischarge']
            m = (q > 0) & (q < 1.3) & (x <= r.life)
            if mode == 'early':
                m &= x <= 100
            qq = np.where(m, q, np.nan)
            ax.plot(
                x[m] if mode == 'early' else x,
                qq[m] if mode == 'early' else qq,
                color=plt.cm.viridis((r.life - 350) / 1650),
                lw=0.7,
                alpha=0.6,
            )
        ax.set(title=f'{b} · n={len(g)}', xlabel='Cycle')
        if mode == 'full':
            ax.axhline(0.88, c='#dd7745', ls='--', lw=1)
            ax.set_xlim(0, 2000)
        else:
            ax.set_xlim(1, 100)
    axes[0].set_ylabel('방전 용량 Qd (Ah)')
    axes[0].set_ylim((0.87, 1.13) if mode == 'full' else (1.0, 1.13))
    save('q2_' + mode)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (b, g) in zip(axes, d.groupby('batch')):
    r = g.sort_values('life').iloc[0]
    s = cells[r.cell]['summary']
    x = s['cycle']
    q = s['QDischarge']
    m = (x >= 20) & (x <= r.life) & (q > 0) & (q < 1.3)
    ax.plot(x[m], q[m], color=COLORS[b], lw=1, label='Qd')
    k = r.knee
    if np.isfinite(k):
        sm = median_filter(q[m], size=21, mode='nearest')
        a = np.column_stack([np.ones(m.sum()), x[m], np.maximum(0, x[m] - k)])
        coef = np.linalg.lstsq(a, sm, rcond=None)[0]
        ax.plot(x[m], a @ coef, color='#182f46', ls='--', label='2구간 회귀')
        ax.axvline(k, color='#dd7745', ls=':', label=f'Knee ≈{k:.0f}')
    ax.set(title=f'{r.cell} · 수명 {r.life:.0f}', xlabel='Cycle')
    ax.legend(fontsize=9)
axes[0].set_ylabel('Qd (Ah)')
save('q2_knees')
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
for b, g in d.groupby('batch'):
    axes[0].scatter(
        g.life,
        g.knee,
        color=COLORS[b],
        label=f'{b}: {g.knee.notna().sum()}/{len(g)}',
        s=25,
        alpha=0.8,
    )
    vals = g.accel_ratio.clip(upper=100)
    axes[1].scatter(g.life, vals, color=COLORS[b], s=25, alpha=0.75)
axes[0].set(xlabel='Cycle Life', ylabel='추정 Knee cycle')
axes[0].legend()
axes[1].set(xlabel='Cycle Life', ylabel='후기 / 초기 기울기 절댓값 비')
axes[1].set_yscale('log')
axes[1].axhline(2, ls='--', c='#888')
save('q2_knee_summary')
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for b, g in d.groupby('batch'):
    r = g.sort_values('life').iloc[0]
    s = cells[r.cell]['summary']
    m = (s['cycle'] <= r.life) & (s['QDischarge'] > 0) & (s['QDischarge'] < 1.3)
    axes[0].plot(s['cycle'][m], s['QDischarge'][m], label=f'{r.cell} ({r.life:.0f})', c=COLORS[b])
    axes[1].plot(s['cycle'][m] / r.life, s['QDischarge'][m] / r.q2, label=r.cell, c=COLORS[b])
axes[0].axhline(0.88, c='#888', ls='--')
axes[0].set(xlabel='Cycle', ylabel='Qd (Ah)')
axes[0].legend()
axes[1].set(xlabel='Cycle / Life', ylabel='Qd / Qd(2)')
axes[1].legend()
save('q1_shortest')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (b, g) in zip(axes, d.groupby('batch')):
    for group, mask, col in [('단수명 <500', g.life < 500, '#dd7745'), ('장수명 >1,000', g.life > 1000, '#167c80')]:
        sub = g[mask & g.dq_valid]
        n = len(sub)
        if n:
            mat = np.array([cells[c]['dq'][::-1] for c in sub.cell])
            v = cells[sub.iloc[0].cell]['v'][::-1]
            med = np.median(mat, axis=0)
            lo, hi = np.quantile(mat, [0.25, 0.75], axis=0)
            ax.plot(v, med, c=col, label=f'{group} n={n}')
            ax.fill_between(v, lo, hi, color=col, alpha=0.18)
        else:
            ax.plot([], [], c=col, label=f'{group} n=0')
    ax.set(title=b, xlabel='Voltage (V)')
    ax.axhline(0, c='#888', lw=0.8)
    ax.legend(fontsize=9, loc='lower left')
axes[0].set_ylabel('ΔQ₁₀₀₋₁₀(V) (Ah)')
save('q3_group_curves')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, (b, g) in zip(axes, d.groupby('batch')):
    rho, _, n = corr(g.dq_logvar.to_numpy(), g.life.to_numpy())
    ax.scatter(g.dq_logvar, g.life, color=COLORS[b], s=27, alpha=0.8)
    ax.set(
        title=f'{b} · Spearman ρ={rho:.3f} · n={n}',
        xlabel='log₁₀ Var(ΔQ) (Ah²)',
        ylabel='Cycle Life',
    )
    ax.set_yscale('log')
save('q3_feature_scatter')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for ax, (b, g) in zip(axes, d.groupby('batch')):
    r = g.sort_values('life').iloc[0]
    c = cells[r.cell]
    ax.plot(c['v'], c['q10'], label='cycle 10', c='#167c80')
    ax.plot(c['v'], c['q100'], label='cycle 100', c='#dd7745')
    ax.set(title=r.cell, xlabel='Voltage (V)', ylabel='Q(V) (Ah)')
    ax.legend()
save('q3_qv')
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
for ax, (b, g) in zip(axes, prot.groupby('batch')):
    g = g.sort_values('mean')
    short = g.policy.str.replace('-newstructure', ' [new]', regex=False)
    yp = np.arange(len(g))
    ax.scatter(g['mean'], yp, c=COLORS[b], s=30)
    for yy, (_, r) in zip(yp, g.iterrows()):
        ax.plot([r['min'], r['max']], [yy, yy], c=COLORS[b], lw=2, alpha=0.4)
        ax.text(r['max'] + 12, yy, f"n={int(r['count'])}", fontsize=8, va='center')
    ax.set_yticks(yp, short, fontsize=8)
    ax.set(title=b, xlabel='평균 수명 (점) / min–max (선)')
    ax.set_xlim(300, 2100)
fig.tight_layout(w_pad=2)
save('q4_protocols')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
for ax, (b, g) in zip(axes, d.groupby('batch')):
    rho, _, n = corr(g.c_eff.to_numpy(), g.life.to_numpy())
    ax.scatter(g.c_eff, g.life, c=COLORS[b], s=30, alpha=0.75)
    ax.set(title=f'{b} · ρ={rho:.3f}', xlabel='0–80% SOC 유효 C-rate')
axes[0].set_ylabel('Cycle Life')
save('q4_crate')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for ax, (b, g) in zip(axes, d.groupby('batch')):
    for name, grp, col in [('최단', g.nsmallest(1, 'life'), '#dd7745'), ('최장', g.nlargest(1, 'life'), '#167c80')]:
        r = grp.iloc[0]
        c = cells[r.cell]['current']
        m = (c['I'] > 0.05) & (c['Qc'] <= 0.88)
        ax.plot(c['t'][m], c['I'][m], c=col, label=f'{name} {r.cell}', lw=1)
    ax.set(title=f'{b} · cycle 10', xlabel='Time (min)', ylabel='충전 C-rate')
    ax.legend(fontsize=8)
save('q4_current_patterns')
qc = pd.DataFrame(q4)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, target, title in zip(axes, ['life', 'late_slope'], ['충전 신호 ↔ 수명', '충전 신호 ↔ 후기 Qd 기울기']):
    pivot = qc[qc.target == target].pivot(index='feature', columns='batch', values='rho').reindex(
        index=['c1', 'c2', 'c_eff', 'i_mean', 'i_std', 'i_high', 'charge_time'],
        columns=['ALL', 'B1', 'B2', 'B3'],
    )
    mat = pivot.to_numpy()
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(4), pivot.columns)
    ax.set_yticks(range(len(pivot)), [LABELS[k] for k in pivot.index], fontsize=9)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f'{mat[i, j]:.2f}' if np.isfinite(mat[i, j]) else '—',
                ha='center',
                va='center',
                fontsize=9,
                color='white' if abs(mat[i, j]) > 0.65 else '#123',
            )
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label='Spearman ρ')
save('q4_correlations')
# Q5: 전체·배치 내부·배치 보정 상관을 나란히 비교해 배치 효과를 확인한다.
order = co[co.batch == 'ALL'].sort_values('spearman', key=abs, ascending=False).feature.tolist()
pivot = co.pivot(index='feature', columns='batch', values='spearman').reindex(
    index=order,
    columns=['ALL', 'B1', 'B2', 'B3'],
)
pivot['배치 보정'] = pa.set_index('feature').batch_adjusted_rank_r
fig, ax = plt.subplots(figsize=(9, 9))
mat = pivot.to_numpy()
im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(5), pivot.columns)
ax.set_yticks(range(len(pivot)), [LABELS[k] for k in pivot.index])
ax.set_title('Q5 | 초기 100 cycles 피처 ↔ Cycle Life')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(
            j,
            i,
            f'{mat[i, j]:.2f}' if np.isfinite(mat[i, j]) else '—',
            ha='center',
            va='center',
            fontsize=9,
            color='white' if abs(mat[i, j]) > 0.65 else '#123',
        )
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, label='Spearman / 배치 보정 순위 상관')
save('q5_target_correlations')
sel = [
    'dq_logvar',
    'dq_logstd',
    'dq_logabsmin',
    'dq_mean',
    'q2',
    'q100',
    'q_ratio',
    'q_slope',
    'ir_mean',
    'ir_delta',
    'tmax_mean',
    'tavg_mean',
    'charge_time',
    'c_eff',
    'i_mean',
]
mat = fc.loc[sel, sel].to_numpy()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(sel)), [LABELS[k] for k in sel], rotation=65, ha='right', fontsize=9)
ax.set_yticks(range(len(sel)), [LABELS[k] for k in sel], fontsize=9)
for i in range(len(sel)):
    for j in range(len(sel)):
        if i != j and abs(mat[i, j]) >= 0.9:
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label='Spearman ρ')
ax.set_title('Q5 | 피처 간 상관: |ρ| ≥0.9만 숫자 표시')
save('q5_collinearity')
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
axes[0].bar(bs.batch, bs.usable, color=[COLORS[b] for b in bs.batch], label='EDA 확정 라벨')
axes[0].bar(bs.batch, bs.excluded, bottom=bs.usable, color='#cbd2db', label='제외 / 검열')
for i, r in bs.iterrows():
    axes[0].text(i, r.usable / 2, str(int(r.usable)), ha='center', color='white')
    axes[0].text(i, r.usable + r.excluded / 2, str(int(r.excluded)), ha='center')
axes[0].set_ylabel('셀 수')
axes[0].legend(fontsize=9)
for b in COLORS:
    g = df[df.batch == b]
    axes[1].scatter(g.n_cycles, g.q_end, c=COLORS[b], label=b, marker='o', s=24, alpha=0.7)
axes[1].axhline(0.885, c='#dd7745', ls='--')
axes[1].set(xlabel='마지막 관측 Cycle', ylabel='마지막 Qd (Ah)', ylim=(0.8, 2.3))
axes[1].legend()
save('data_quality')
stats = {
    'batch_summary': summary,
    'n_raw': len(df),
    'n_usable': len(d),
    'extra': audit_extra,
    'dq_ci': ci,
    'kruskal': {
        'stat': float(kruskal(*[g.life.to_numpy() for _, g in d.groupby('batch')]).statistic),
        'p': float(kruskal(*[g.life.to_numpy() for _, g in d.groupby('batch')]).pvalue),
    },
    'dq_coverage': int(d.dq_valid.sum()),
    'feature_complete': int(d[FEATURES].notna().all(axis=1).sum()),
    'dq_shortest_removed': corr(d[d.life > d.life.min()].dq_logvar.to_numpy(), d[d.life > d.life.min()].life.to_numpy())[0],
    'vif': vifrows,
    'collinearity_pairs': pairs,
    'q_ratio_gain_pct': float((d.q_ratio > 1).mean() * 100),
    'knee_n': int(d.knee.notna().sum()),
    'knee_smoothing_median_span': float(d.knee_smoothing_span.median()),
}
with open(ROOT / 'statistics.json', 'w') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=lambda x: float(x))
print(bs.to_string(index=False), flush=True)
print(
    'Top correlations',
    co[co.batch == 'ALL'].sort_values('spearman', key=abs, ascending=False).head(10).to_string(
        index=False,
    ),
    flush=True,
)
print('All early features missing:', d[FEATURES].isna().sum().to_dict(), flush=True)
print('Knees:', stats['knee_n'], 'CI', ci, flush=True)
