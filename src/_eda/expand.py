"""기본 EDA에 누적분포·정규화 열화·전압 구간·태그 분석을 추가한다.

extract.py의 결과에서 보충 표와 그래프를 만든다.
상대 수명 정규화와 후기 기울기는 사후 EDA용이며 모델 입력이 아니다."""
from pathlib import Path
import os
import pickle
import json
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / '.cache/preprocessing'
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
from scipy.ndimage import median_filter
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
        'axes.grid': True,
        'grid.alpha': 0.17,
    },
)
COL = {'B1': '#167c80', 'B2': '#dd7745', 'B3': '#6162aa'}
D = pd.read_csv(ROOT / 'features.csv')
with open(ROOT / 'extracted.pkl', 'rb') as f:
    cells = pickle.load(f)


def save(name):
    """현재 보충 분석 그림을 저장하고 닫는다."""
    plt.savefig(PROJECT / 'results/figures' / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for b, g in D.groupby('batch'):
    x = np.sort(g.life)
    axes[0].step(
        x,
        np.arange(1, len(x) + 1) / len(x),
        where='post',
        color=COL[b],
        label=f'{b} n={len(g)}',
    )
for label, value, col in [('태그 없음', False, '#dd7745'), ('[new]', True, '#167c80')]:
    g = D[(D.batch == 'B2') & (D.newstructure == value)]
    x = np.sort(g.life)
    axes[1].step(
        x,
        np.arange(1, len(x) + 1) / len(x),
        where='post',
        color=col,
        label=f'{label} n={len(g)}',
    )
for ax in axes:
    ax.set(xlabel='Cycle Life', ylabel='누적 비율', ylim=(0, 1.03))
    ax.legend()
    ax.axvline(500, ls=':', c='#999')
    ax.axvline(1000, ls=':', c='#999')
axes[0].set_title('배치별 경험적 누적분포')
axes[1].set_title('B2: 동일 배치 내부의 두 집단')
save('q1_ecdf')
q1 = []
for b, g in D.groupby('batch'):
    q1.append(
        {
            'batch': b,
            'below_800_pct': float((g.life <= 800).mean() * 100),
            'cv_pct': float(g.life.std() / g.life.mean() * 100),
            'skew': float(g.life.skew()),
        },
    )
stages = []
matrices = {}
# 수명이 다른 셀의 전체 열화 진행을 상대 사이클로 맞춘다. 실제 수명을 쓰므로 EDA에만 사용한다.
grid = np.linspace(0.02, 0.98, 193)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for b, g in D.groupby('batch'):
    mat = []
    for r in g.itertuples():
        s = cells[r.cell]['summary']
        x = s['cycle']
        q = s['QDischarge']
        m = (x >= 2) & (x <= r.life) & (q > 0) & (q < 1.3)
        xx = x[m] / r.life
        yy = median_filter(q[m], size=11, mode='nearest') / r.q2
        mat.append(np.interp(grid, xx, yy))
    mat = np.array(mat)
    med = np.median(mat, axis=0)
    lo, hi = np.quantile(mat, [0.25, 0.75], axis=0)
    axes[0].plot(grid, med, c=COL[b], label=b)
    axes[0].fill_between(grid, lo, hi, color=COL[b], alpha=0.14)
    stages.append(
        {
            'batch': b,
            'n': len(g),
            'q2_median': float(g.q2.median()),
            'q100_median': float(g.q100.median()),
            'growth_median_pct': float((g.q_ratio.median() - 1) * 100),
            'early_slope_median_mah': float(g.q_slope.median() * 1000),
            'pre_knee_median_mah': float(g.early_slope.median() * 1000),
            'post_knee_median_mah': float(g.late_slope.median() * 1000),
            'knee_cycle_median': float(g.knee.median()),
            'retention_50': float(np.median(mat[:, np.argmin(abs(grid - 0.5))]) * 100),
            'retention_75': float(np.median(mat[:, np.argmin(abs(grid - 0.75))]) * 100),
            'retention_90': float(np.median(mat[:, np.argmin(abs(grid - 0.9))]) * 100),
        },
    )
axes[0].set(xlabel='Cycle / Life', ylabel='Qd / Qd(2)', title='상대 수명에 따른 용량 유지율')
axes[0].legend()
axes[0].axhline(1, c='#888', ls=':')
axes[1].boxplot([(g.q_ratio - 1) * 100 for _, g in D.groupby('batch')], tick_labels=list(COL))
for j, (b, g) in enumerate(D.groupby('batch'), 1):
    axes[1].scatter(np.repeat(j, len(g)), (g.q_ratio - 1) * 100, c=COL[b], s=12, alpha=0.5)
axes[1].axhline(0, c='#888', ls=':')
axes[1].set(ylabel='Qd(100)/Qd(2) - 1 (%)', title='초기 100사이클의 용량 증감')
save('q2_normalized')
# ΔQ 변화를 전압 구간별로 나누어 어떤 영역이 수명과 더 관련되는지 확인한다.
bands = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.3), (3.3, 3.5)]
reg = []
groups = []
for r in D.itertuples():
    c = cells[r.cell]
    for lo, hi in bands:
        m = (c['v'] >= lo) & (c['v'] < hi if hi < 3.5 else c['v'] <= hi)
        var = np.var(c['dq'][m], ddof=1)
        reg.append(
            {
                'cell': r.cell,
                'batch': r.batch,
                'life': r.life,
                'voltage': f'{lo:.1f}–{hi:.1f} V',
                'logvar': np.log10(max(var, 1e-20)),
            },
        )
for b, g in D.groupby('batch'):
    low, high = g.life.quantile([0.25, 0.75])
    for name, sub in [('하위 25%', g[g.life <= low]), ('상위 25%', g[g.life >= high])]:
        points = np.array([np.interp(3.0, cells[c]['v'][::-1], cells[c]['dq'][::-1]) for c in sub.cell])
        groups.append(
            {
                'batch': b,
                'group': name,
                'n': len(sub),
                'life_median': float(sub.life.median()),
                'logvar_median': float(sub.dq_logvar.median()),
                'delta_at_3v_mah': float(np.median(points) * 1000),
                'min_dq_median_mah': float(np.median([np.min(cells[c]['dq']) for c in sub.cell]) * 1000),
                'mean_dq_median_mah': float(sub.dq_mean.median() * 1000),
            },
        )
R = pd.DataFrame(reg)
corr = []
for b, g in [('ALL', R), *list(R.groupby('batch'))]:
    for voltage, z in g.groupby('voltage'):
        corr.append(
            {
                'batch': b,
                'voltage': voltage,
                'rho': float(spearmanr(z.logvar, z.life).statistic),
                'n': len(z),
            },
        )
T = pd.DataFrame(corr).pivot(index='voltage', columns='batch', values='rho').reindex(
    columns=['ALL', 'B1', 'B2', 'B3'],
)
fig, ax = plt.subplots(figsize=(9, 3))
a = T.to_numpy()
im = ax.imshow(a, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(4), T.columns)
ax.set_yticks(range(4), T.index)
ax.set_title('전압 구간별 log Var(ΔQ) ↔ 수명')
for i in range(4):
    for j in range(4):
        ax.text(
            j,
            i,
            f'{a[i, j]:.3f}',
            ha='center',
            va='center',
            color='white' if abs(a[i, j]) > 0.65 else '#123',
        )
fig.colorbar(im, ax=ax, pad=0.03, label='Spearman ρ')
save('q3_voltage_regions')
# 같은 숫자 충전 정책도 태그에 따라 결과가 다를 수 있다. 관측 차이를 인과 효과로 해석하지 않는다.
G = D[D.batch == 'B2']
q4 = []
for name, g in [('B2 전체', G), ('B2 태그 없음', G[~G.newstructure]), ('B2 [new]', G[G.newstructure])]:
    q4.append(
        {
            'group': name,
            'n': len(g),
            'life_mean': float(g.life.mean()),
            'life_median': float(g.life.median()),
            'rho_ceff': float(spearmanr(g.c_eff, g.life).statistic),
            'rho_imean': float(spearmanr(g.i_mean, g.life).statistic),
        },
    )
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for name, value, col in [('태그 없음', False, '#dd7745'), ('[new]', True, '#167c80')]:
    g = G[G.newstructure == value]
    axes[0].scatter(g.i_mean, g.life, c=col, label=f'{name} n={len(g)}', s=30)
axes[0].set(xlabel='실측 평균 C-rate', ylabel='Cycle Life', title='B2: 태그를 나누어 본 충전–수명')
axes[0].legend()
policies = ['4.8C(80%)-4.8C', '5.2C(58%)-4C', '5.6C(26%)-4.5C']
yp = np.arange(3)
for label, value, col, offset in [('태그 없음', False, '#dd7745', -0.15), ('[new]', True, '#167c80', 0.15)]:
    vals = [G[(G.policy.str.replace('-newstructure', '', regex=False) == p) & (G.newstructure == value)].life.mean(
    ) for p in policies]
    axes[1].barh(yp + offset, vals, height=0.28, color=col, label=label)
    for y, v in zip(yp + offset, vals):
        axes[1].text(v + 7, y, f'{v:.1f}', va='center', fontsize=9)
axes[1].set_yticks(yp, policies, fontsize=9)
axes[1].set(xlabel='정책별 평균 수명', title='같은 숫자 정책, 다른 태그')
axes[1].set_xlim(0, 1150)
axes[1].legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, fontsize=9)
fig.tight_layout()
save('q4_tag_control')
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for b, g in D.groupby('batch'):
    axes[0].scatter(g.q2, g.life, c=COL[b], label=b, s=25, alpha=0.75)
    ax = axes[1]
    rankx = G
# Q2와 수명의 전체 상관이 배치별 평균 차이에서 생기는지 순위의 배치 평균을 제거해 확인한다.
rank = D[['q2', 'life', 'batch']].copy()
rank['qrank'] = rank.q2.rank()
rank['yrank'] = rank.life.rank()
rank['qr'] = rank.qrank - rank.groupby('batch').qrank.transform('mean')
rank['yr'] = rank.yrank - rank.groupby('batch').yrank.transform('mean')
for b, g in rank.groupby('batch'):
    axes[1].scatter(g.qr, g.yr, c=COL[b], s=25, alpha=0.75)
axes[0].set(xlabel='Qd(2) (Ah)', ylabel='Cycle Life', title='전체 데이터: 초기 용량과 수명')
axes[0].legend()
axes[1].set(
    xlabel='Qd(2) 순위 - 배치 평균 순위',
    ylabel='수명 순위 - 배치 평균 순위',
    title='배치 차이 제거 후: 관계가 거의 사라짐',
)
axes[1].axhline(0, c='#aaa', lw=0.7)
axes[1].axvline(0, c='#aaa', lw=0.7)
save('q5_batch_confounding')
pd.DataFrame(stages).to_csv(ROOT / 'degradation_stage_summary.csv', index=False)
pd.DataFrame(groups).to_csv(ROOT / 'dq_group_details.csv', index=False)
pd.DataFrame(corr).to_csv(ROOT / 'voltage_region_correlations.csv', index=False)
pd.DataFrame(q4).to_csv(ROOT / 'charge_tag_correlations.csv', index=False)
json.dump(
    {'q1': q1, 'q2': stages, 'q3_groups': groups, 'q3_regions': corr, 'q4_tag': q4},
    open(ROOT / 'expanded_statistics.json', 'w'),
    ensure_ascii=False,
    indent=2,
)
print(
    json.dumps(
        {'q1': q1, 'q2': stages, 'q3_groups': groups, 'q3_regions': corr, 'q4_tag': q4},
        ensure_ascii=False,
        indent=2,
    ),
)
