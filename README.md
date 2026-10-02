# ESS 배터리 수명 예측

초기 100사이클의 측정값으로 최종 **Cycle Life**를 예측하고, 다른 배치로 일반화되는지 평가한다. ESS의 점검·교체 계획 검토에 활용할 수 있는 수명 신호를 탐색한다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터: Batch 1 (2017-05-12), 36셀
- 평가 데이터: Batch 2 (2018-02-20), 39셀
- 추가 평가: Batch 3 (2018-04-12), 43셀
- 태스크: **Regression — Cycle Life 예측**
- 예측 시점: cycle 100, 타깃: log₁₀(Cycle Life), 역변환: 10**예측값

원시 139셀 중 유효 수명 라벨과 수집 품질 기준을 만족한 118셀을 사용했다.

## 파일 구조

```text
├── data/
│   ├── README.md
│   └── raw/                         원본 MAT 보관 위치 — Git 제외
├── notebooks/
│   ├── 01_EDA.ipynb                 원본 MAT → Q1–Q5 직접 계산·시각화
│   ├── 02_feature_engineering.ipynb  초기 피처·전처리·누수 방지
│   └── 03_modeling.ipynb            모델 선택·성능·오류 분석
├── src/
│   ├── preprocess.py               데이터 로딩 및 원본 재추출
│   ├── features.py                 명시적 초기 피처 생성
│   ├── train.py                    B1 모델 선택 및 B2/B3 평가
│   └── _eda/                       원본 EDA 재현 코드
├── results/
│   ├── figures/                     EDA·평가 그래프
│   └── model_performance.csv        최종 성능 보고 표
├── reports/
│   └── DS-MINI-Design-*.pdf          EDA 및 모델 전략 보고서
├── config.json
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## 환경 설정

Python **3.11.11**과 **uv**를 사용한다. 의존성은 `pyproject.toml`과 `uv.lock`으로 관리한다. 저장소를 내려받은 디렉토리에서 실행한다.

```bash
git clone https://github.com/www-jong/SKALA_DS_Mini_Project.git
cd SKALA_DS_Mini_Project
uv python install 3.11.11
uv sync --locked
uv run python -m ipykernel install --user --name skala-ds --display-name "SKALA DS (uv · Python 3.11.11)"
uv run jupyter lab
```

uv가 없다면 [uv 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 먼저 설치한다. macOS에서는 LightGBM의 OpenMP 실행 라이브러리를 준비한다.

```bash
brew install libomp
```

Windows에서는 Visual C++ Runtime이 필요할 수 있다. 시스템 의존성은 [LightGBM 공식 설치 안내](https://github.com/lightgbm-org/LightGBM/tree/main/python-package)를 참고한다. 전체 실행 검증은 macOS Apple Silicon의 새 uv 가상환경에서 수행했다.

위 명령으로 환경을 준비한 뒤 노트북을 실행한다. Jupyter에서는 **SKALA DS (uv · Python 3.11.11)** 커널을 선택한다. 커널 등록은 이 저장소의 `.venv` Python을 사용하도록 연결하는 과정이다. 저장소를 다른 경로로 옮기거나 환경을 다시 만들면 등록 명령도 다시 실행한다.

`.python-version`은 uv가 사용할 Python 버전(3.11.11)을 지정하는 설정 파일이다. `pyproject.toml`은 허용 버전과 의존성을, `uv.lock`은 설치할 의존성의 정확한 버전을 기록한다. 세 파일 모두 재현을 위해 Git에 포함한다.

**clone만으로 원본 데이터까지 내려받아지지는 않는다.** 아래 파일을 `data/raw`에 직접 준비한 뒤 노트북을 실행한다.

`data/raw`의 MAT 4개 중 Batch 1·2·3을 분석한다. `2018-04-03_varcharge`는 별도 실험 파일로 보관한다. `notebooks`의 01 → 02 → 03 순서로 실행한다. 노트북에 실행 결과가 포함되어 GitHub에서도 그래프를 확인할 수 있다. EDA 노트북 재실행에는 원본 MAT 3개가 필요하다. `data/raw`에 준비하거나 `BATTERY_DATA_DIR`로 디렉토리를 지정한다. 피처 생성도 같은 원본을 사용하며, 원본 파일이 바뀌거나 중간 산출물이 없으면 다시 추출한다. 중간 표·피처·후보 비교·학습 모델은 Git에서 제외되는 `.cache`에 자동 생성한다. `results`에는 그래프와 최종 성능표만 저장한다. 원본 MAT는 업로드하지 않는다.

```bash
uv run python -m src.features
uv run python -m src.train
# 원본 재추출이 필요한 경우, MAT를 data/raw에 준비한다.
uv run python -m src.preprocess --rebuild --dataset data/raw
```

모델링 노트북은 완료된 결과가 없거나 원본·설정·실행 코드·결과 파일이 변경되면 후보 비교·학습을 실행한다. 입력과 결과가 일치하는 완료된 실험만 재사용한다. 전체 모델 비교를 다시 학습하려면 `uv run python -m src.train`을 실행한다. 코드 역할과 학습 흐름은 [src/README.md](src/README.md), 데이터 취득 및 파일 설명은 [data/README.md](data/README.md)를 참고한다.

## EDA

- **Cycle Life 분포:** 중앙값은 B1 772.5, B2 472, B3 1,002사이클. 단수명(<500) 28셀은 모두 B2이며, B1 학습에는 단수명 외삽 위험이 있다.
- **열화 곡선:** 초기 용량이 증가한 셀은 93/118개. 후기 감소가 가속되며 탐색된 Knee의 상대 시점은 배치별 중앙값 약 75–79%. Knee와 후기 지표는 모델 입력에서 제외한다.
- **ΔQ(V):** Q100−Q10의 log 분산과 수명의 Spearman 상관은 −0.885. 배치 내부에서도 강한 관계를 유지해 기준 피처로 채택했다.
- **충전 속도:** 최대 C-rate만으로 수명을 설명할 수 없다. 전환 SOC와 동일 숫자 정책의 태그·배치 차이를 함께 검토했다.
- **상관·공선성:** 초기 용량 Q2의 전체 상관은 배치 보정 후 약해졌다. log 표준편차와 log 분산은 완전 중복이므로 동시에 사용하지 않는다.

자세한 그래프와 해석: [01_EDA.ipynb](notebooks/01_EDA.ipynb). 원본 MAT 로딩부터 셀별 데이터 가공, `groupby`·통계 계산·Matplotlib 시각화까지 노트북 셀에 직접 구현했다. 저장된 분석 JSON·CSV·PNG나 외부 분석 모듈을 불러오지 않는다. 보고서의 그림 1–23을 노트북에서 새로 생성하며 Q2의 전체 궤적·초기 확대·Knee 사례·Knee 분포·정규화 궤적(그림 6–10)을 모두 포함한다. 원본 MAT 재추출에서 수치 테이블 13개와 그래프 23개의 재현을 확인했다.

## Modeling

### 피처 엔지니어링 전략

DAY 1에서 설계한 후보는 `dq_logvar`, `q2`, `q_slope_last`, `ir_delta`, `tmax_mean`, `i_high`의 Core6다. [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb)에서 각 피처의 EDA 근거·정의·단위·추가 가치 검증 기준을 설명하고 원본 MAT부터 직접 계산한다. ΔQ 기준선 → 용량 추가 → 저항·온도 추가 → 충전 패턴 추가의 비교군을 구성하며, 논문 방전 6피처는 별도 비교군으로 둔다. 설계 후보와 03에서의 최종 선정 결과를 구분한다. ΔQ 단일 → 용량 3개 → 용량·저항·온도 5개 → Core6의 단계별 비교와 논문 방전 6개 피처군 비교를 수행한다.

초기 100사이클의 ΔQ·용량·저항·온도·충전 피처 73개를 구성했다. 현재 모델 비교에서는 명시적으로 정의한 5개 피처군만 사용하며, 모든 후보를 일괄 투입하지 않는다.

최종 6피처는 log|min ΔQ|, log 분산, log|왜도|, log 첨도, Q2, max(Q2..100)−Q2다. 최대 용량을 전체 수명에서 구하지 않는다. 왜도·첨도는 보충자료의 수식과 정의를 확인했다. 대치·표준화는 학습 fold에서만 fit한다.

### EDA 전략의 구현 및 검증

모델 개발의 기준은 EDA에서 세운 전략을 구현하고, 그 범위에서 후보를 비교하여 최종 모델을 선정하는 것이다. 논문 성능은 비교값이며, 점수 차이와 오류 분석으로 전략의 한계를 평가한다.

| EDA 관찰                               | 개발 전략                                           | 구현·검증                                    |
| -------------------------------------- | --------------------------------------------------- | --------------------------------------------- |
| ΔQ 분산이 배치 내부에서도 수명과 관련 | 단일 ΔQ 기준선과 추가 통계 피처 비교               | ΔQ 단일·용량 3개·용량/저항/온도 5개·Core6·논문 방전 6개 비교         |
| 작은 B1 표본과 피처 공선성             | 정규화 모델을 우선 비교하고 복잡한 후보의 기여 확인 | 11개 모델을 동일한 B1 내부 평가로 비교        |
| 초기 용량 증가와 후기 가속 열화        | 초기 100사이클의 증가량만 사용                      | Knee·후기 기울기·최종 용량은 입력에서 제외  |
| 충전 정책 반복 및 배치 차이            | 동일 정책을 그룹으로 묶고 외부 배치를 분리          | 프로토콜 단위 CV·Hold-out, B2/B3 최종 평가   |
| 배치별 피처 관계 차이                  | 테스트 점수로 후보를 선택하지 않고 잔차 진단        | B1에서 모델 확정 후 배치별 편향·큰 오차 분석 |

### 모델 선택 및 근거

- 후보: Linear, Ridge, ElasticNet, Huber, BayesianRidge, SVR, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost
- 비교 범위: 11모델 × 5피처군 = 55설정, 제한된 파라미터 탐색 · 실패 후보 0개
- 최종 모델: **BayesianRidge + 논문 방전 6피처**
- 선택 근거: B1 개발 26셀의 프로토콜 분리 nested 4-fold CV에서 평균 MAPE가 가장 낮았다. 내부 3-fold에서 파라미터를 튜닝했다.

피처군별 최선 후보의 개발 CV 평균 MAPE는 ΔQ 단일 9.87%, 용량 3개 6.43%, 용량·저항·온도 5개 8.44%, Core6 8.60%, 논문 방전 6개 6.28%였다. 피처군마다 최선 모델이 다르므로, 동일 모델에서 피처 하나의 순수 효과를 측정한 비교는 아니다. 현재 후보 범위에서는 용량 피처 추가가 기준선보다 유용했고 저항·온도·충전 신호 추가는 성능 개선으로 이어지지 않았다.

Hold-out 10셀의 충전 프로토콜은 개발군과 겹치지 않는다. Hold-out 및 B2/B3 점수로 모델을 선택하지 않았다. 선택 후 파라미터를 고정하고 전체 B1 36셀로 재학습하여 외부 배치를 평가했다.

“최적”은 현재 후보와 탐색 범위 내의 최적이다. 모든 배치를 이미 EDA와 이전 탐색에서 확인했으므로 새로운 블라인드 테스트라고 주장하지 않는다.

## 성능 결과

| 구분                         | MAPE (%) / Gap (pp) | 비고                                           |
| ---------------------------- | ------------------: | ---------------------------------------------- |
| Train (Batch 1 CV)           |                5.33 | B1 36셀 · 프로토콜 분리 nested5fold MAPE 평균 |
| Valid (Batch 1 Hold-out)     |                5.13 | 개발26셀로 학습 · 독립 프로토콜10셀 검증      |
| Test (Batch 2)               |               23.71 | B1 36셀 재학습 후 B2 39셀 평가                 |
| Gap (Train-Valid)            |               -0.20 | Valid−Train · pp · (+) 과적합 의심          |
| Gap (Valid-Test)             |               18.58 | B2−Valid · pp · (+) 배치 일반화 저하        |
| Gap (Target-Test)            |               14.61 | B2−9.1% · pp                                 |
| Test (Batch 3)               |               10.36 | 같은 확정 모델 · B3 43셀 평가                 |
| Gap (Batch2-Batch3)          |              -13.35 | B3−B2 · pp · (+) B3 성능 저하               |
| Gap (Target-Test) — Batch 3 |                1.26 | B3−9.1% · pp                                 |

Gap은 양수일 때 오차 증가를 나타내도록 계산했다. Train-Valid=Valid−Train, Valid-Test=Test−Valid, Target-Test=Test−9.1, Batch2-Batch3=B3−B2이며 단위는 퍼센트포인트(pp)다.

Train은 B1 전체 36셀의 nested 5-fold 평균이다. B1 개발군에서 후보를 선택한 뒤의 보고용 CV이므로 완전히 독립적인 모델 선택 평가로 해석하지 않는다. Valid는 개발 26셀 모델, Test는 B1 전체 재학습 모델의 점수이므로 학습량 차이도 있다. Gap을 과적합의 확정 판정으로 해석하지 않는다.

Target 9.1%는 지정된 비교값이며 B2·B3 모두 이 값과의 차이를 보고한다. 보조 MAE·RMSE와 셀별 잔차는 모델링 노트북의 결과와 오류 분석에서 확인한다.

## 오류 분석

B2에서 평균 +128.9사이클의 과대예측이 관측됐다. B1에 없던 단수명으로의 외삽과 배치별 피처 관계 차이가 가능한 원인이다. B2-C45(841→1549), C10(791→1289)은 중간 수명에서도 크게 틀려 ΔQ 형태·프로토콜·수집 품질을 추가 확인해야 한다. 큰 오차를 이유로 셀을 삭제하지 않았다.

B3 MAPE는 10.36%로 B2보다 좋았다. B3가 항상 더 어렵다는 가정은 현재 모델의 관측 결과와 맞지 않는다. 단수명 학습 데이터 확보, 정책 외 검증, 배치 편향과 큰 잔차를 추가 검토한다. 자세한 분석: [03_modeling.ipynb](notebooks/03_modeling.ipynb).

## ESS 도메인 해석

검증된 예측은 점검 우선순위, 교체 계획 검토, 운전정책 비교를 지원할 수 있다. 본 모델은 고속 충전 LFP/graphite 셀의 Cycle Life를 예측하며 실제 BESS의 잔여 연수, 안전성, 고장 확률을 직접 예측하지 않는다.

실 배포 전에는 현장 외부 검증, 달력 열화, 운전 온도·SOC·휴지, 팩 불균형, 불확실성 평가와 모니터링이 추가로 필요하다.

## 참고문헌

- Severson et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383–391. [DOI](https://doi.org/10.1038/s41560-019-0356-8)
- [프로젝트 요구사항](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb)

## 팀 구성

- 울산 1반 U018 원종현: EDA, 피처 엔지니어링, 모델 개발, Batch 2·3 성능 평가 및 보고서 작성
