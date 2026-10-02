# src 사용 안내

`src`는 전처리, 피처 생성, 모델 학습을 실행하는 코드입니다. 01·02 노트북은 계산 과정을 셀에 직접 구현하고, 03 노트북은 `src.train`으로 학습·평가를 실행합니다.

| 파일 | 담당하는 일 | 입력 → 출력 |
| --- | --- | --- |
| `preprocess.py` | 원본 확인, 셀 추출, 중간 데이터 저장·로드 | `data/raw` MAT → `.cache/preprocessing` |
| `features.py` | 초기 100사이클로 셀별 후보 피처 계산 | 전처리 표·배열 → `.cache/features` |
| `train.py` | B1 내부 후보 비교, Hold-out 검증, 최종 학습, B2·B3 평가 | 피처 표 → `.cache/modeling`, `results/model_performance.csv` |
| `_eda/extract.py` | MAT 읽기, 채택·제외 기록, 기본 피처와 EDA 통계·그림 생성 | 원본 MAT → 전처리 표·셀 배열·그림 |
| `_eda/supplement.py` | 배치 내부 수명 사분위의 ΔQ 비교, Core6 VIF 계산 | 추출 결과 → 추가 통계·그림 |
| `_eda/expand.py` | 상대 수명 열화, 전압 구간, 충전 태그, 배치 효과 분석 | 추출 결과 → 추가 통계·그림 |

`preprocess.rebuild()`는 `_eda`의 세 스크립트를 순서대로 실행합니다.

## 실행 방법

저장소 루트에서 실행합니다.

```bash
# 전처리: 필요한 중간 파일이 없거나 원본이 변경되면 재추출
uv run python -m src.preprocess

# 원본부터 전처리와 EDA를 다시 실행
uv run python -m src.preprocess --rebuild

# 피처 생성
uv run python -m src.features

# 모델 비교·학습·평가
uv run python -m src.train
```

`features`와 `train`은 필요한 전처리 파일이 없으면 원본부터 처리합니다. 총 73개 후보 피처를 생성하며, 학습에는 지정한 피처군만 사용합니다. 결측 대치와 표준화는 Pipeline에서 학습 fold에만 적합합니다.

중간 데이터는 `.cache/preprocessing`, 피처는 `.cache/features`, 모델·예측·실험 기록은 `.cache/modeling`에 저장합니다. 제출용 그래프와 성능표는 `results`에 저장합니다.

## 학습 흐름

1. B1의 같은 충전 프로토콜을 묶어서 개발군과 Hold-out을 분리합니다.
2. 개발군에서 내부 3-fold 파라미터 탐색과 외부 4-fold 평가로 후보를 비교합니다.
3. 개발 CV의 평균 MAPE로 모델·피처군을 고정한 뒤 Hold-out을 평가합니다.
4. 선택한 모델·피처군의 B1 전체 5-fold CV 성능을 보고합니다.
5. 개발군에서 선택한 파라미터로 B1 전체를 재학습하고 같은 모델로 B2·B3을 평가합니다.

모델 입력은 초기 100사이클의 관측으로 구성합니다. Knee·후기 열화 속도와 배치·셀 ID·실제 수명은 입력에서 제외합니다.
