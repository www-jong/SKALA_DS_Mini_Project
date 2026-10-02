"""배치 내부 수명 사분위의 ΔQ 곡선과 Core6 VIF를 계산한다.

extract.py가 만든 features.csv와 셀 배열을 입력으로 사용한다.
재추출 순서는 extract → supplement → expand이며 독립 로더가 아니다."""
from pathlib import Path
import os
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / '.cache/preprocessing'
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
_font = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
if _font.exists():
    font_manager.fontManager.addfont(str(_font))
COLORS = {'B1': '#167c80', 'B2': '#dd7745', 'B3': '#6162aa'}


def save(name):
    """배치 사분위 비교 그림을 저장하고 닫는다."""
    plt.savefig(PROJECT / 'results/figures' / f'{name}.png', dpi=190, bbox_inches='tight')
    plt.close()
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
D = pd.read_csv(ROOT / 'features.csv')
with open(ROOT / 'extracted.pkl', 'rb') as f:
    cells = pickle.load(f)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (b, g) in zip(axes, D.groupby('batch')):
    # 배치마다 수명 분포가 달라, 고정 500/1,000 기준과 별도로 배치 내부 상·하위 25%를 비교한다.
    lo, hi = g.life.quantile([0.25, 0.75])
    for label, sub, col in [('하위 25%', g[g.life <= lo], '#dd7745'), ('상위 25%', g[g.life >= hi], '#167c80')]:
        mat = np.array([cells[c]['dq'][::-1] for c in sub.cell])
        v = cells[sub.iloc[0].cell]['v'][::-1]
        med = np.median(mat, axis=0)
        low, high = np.quantile(mat, [0.25, 0.75], axis=0)
        ax.plot(v, med, c=col, label=f'{label} n={len(sub)}')
        ax.fill_between(v, low, high, color=col, alpha=0.18)
    ax.set(title=f'{b} · 수명 Q25={lo:.0f}, Q75={hi:.0f}', xlabel='Voltage (V)')
    ax.legend(fontsize=9, loc='lower left')
axes[0].set_ylabel('ΔQ100-10(V) (Ah)')
save('q3_relative_groups')
compact = ['dq_logvar', 'q2', 'q_slope_last', 'ir_delta', 'tmax_mean', 'i_high']
# Core6의 공선성 진단에는 완전 관측 셀을 사용한다. 모델에서 결측 셀을 제거한다는 뜻은 아니다.
z = D[compact].dropna()
z = (z - z.mean()) / z.std()
arr = z.to_numpy()
out = []
for j, k in enumerate(compact):
    a = np.delete(arr, j, axis=1)
    pred = a @ np.linalg.lstsq(a, arr[:, j], rcond=None)[0]
    r2 = 1 - np.sum((arr[:, j] - pred) ** 2) / np.sum(arr[:, j] ** 2)
    out.append({'feature': k, 'vif': 1 / (1 - r2), 'n': len(z)})
pd.DataFrame(out).to_csv(ROOT / 'vif_compact.csv', index=False)
