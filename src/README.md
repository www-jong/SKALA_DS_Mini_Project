# src 사용 안내

노트북은 분석 과정과 근거를 보여 주고, `src`는 같은 처리를 함수나 명령어로 실행할 때 사용한다. `01_EDA.ipynb`와 `02_feature_engineering.ipynb`는 계산 코드를 셀에 직접 구현하며, `03_modeling.ipynb`는 모델링 실행에 `src.train`을 사용한다. 완료 기록과 원본·코드·설정·결과 파일을 확인해 유효한 결과만 재사용한다.

| 파일 | 담당하는 일 | 입력 → 출력 |
| --- | --- | --- |
| `preprocess.py` | 원본 확인, 셀 추출, 중간 데이터 저장·로드 | `data/raw` MAT → `.cache/preprocessing` |
| `features.py` | 초기 100사이클로 셀별 후보 피처 계산 | 전처리 표·배열 → `.cache/features` |
| `train.py` | B1 내부 후보 비교, Hold-out 검증, 최종 학습, B2·B3 평가 | 피처 표 → `.cache/modeling`, `results/model_performance.csv` |
| `_eda/extract.py` | MAT 읽기, 채택·제외 기록, 기본 피처와 EDA 통계·그림 생성 | 원본 MAT → 전처리 표·셀 배열·그림 |
| `_eda/supplement.py` | 배치 내부 수명 사분위의 ΔQ 비교, Core6 VIF 계산 | 추출 결과 → 추가 통계·그림 |
| `_eda/expand.py` | 상대 수명 열화, 전압 구간, 충전 태그, 배치 효과 분석 | 추출 결과 → 추가 통계·그림 |

`_eda`의 세 스크립트는 `preprocess.rebuild()`가 순서대로 실행한다. 각각 따로 실행하려면 앞 단계의 결과가 있어야 한다. `__init__.py`는 `src`를 Python 패키지로 인식시키는 파일이다.

## 실행 방법

저장소 루트에서 실행한다.

```bash
# 원본과 필수 중간 파일 확인. 필요할 때만 다시 추출한다.
uv run python -m src.preprocess

# 원본부터 전처리 통계와 EDA 그림을 다시 만든다.
uv run python -m src.preprocess --rebuild

# 초기 후보 피처를 다시 계산한다.
uv run python -m src.features

# 후보 비교부터 최종 평가까지 실행한다.
uv run python -m src.train
```

`features`와 `train`도 필요한 전처리 파일이 없으면 원본부터 준비한다. 피처는 현재 73개 후보를 만들며, 학습에서는 선택한 피처군만 사용한다. 결측 대치·표준화는 `train.py`의 Pipeline이 학습 fold 안에서 적합한다.

현재 중간 파일의 위치는 `.cache`이며 Git에서 제외한다. 모델·예측·실험 기록은 `.cache/modeling`에, 제출용 성능표와 그림은 `results`에 저장한다. `--rebuild`는 EDA 그림도 다시 저장한다.

## 학습 흐름

1. B1의 같은 충전 프로토콜을 묶어서 개발군과 Hold-out을 분리한다.
2. 개발군에서 내부 3-fold 파라미터 탐색과 외부 4-fold 평가로 후보를 비교한다.
3. 개발 CV의 평균 MAPE로 모델·피처군을 고정한 뒤 Hold-out을 평가한다.
4. 선택한 모델·피처군의 B1 전체 5-fold CV 성능을 보고한다.
5. 개발군에서 선택한 파라미터로 B1 전체를 재학습하고 같은 모델로 B2·B3을 평가한다.

전체 수명으로 구한 Knee나 후기 열화 속도는 EDA 설명용이다. 모델 피처에는 초기 관측만 넣으며, 배치·셀 ID·실제 수명은 입력에서 제외한다.
