# 원본 데이터

`raw/`에 MIT–Stanford Battery Dataset의 MAT 파일을 보관합니다.

| 파일                                                           | 사용 구분            |
| -------------------------------------------------------------- | -------------------- |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat           | Batch 1 · 학습      |
| 2018-02-20_batchdata_updated_struct_errorcorrect.mat           | Batch 2 · 평가      |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat           | Batch 3 · 추가 평가 |
| 2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat | 별도 실험 · 보관    |

원본 출처: [Kaggle - MIT - Stanford Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle).
Batch 1·2·3 파일을 원래 파일명 그대로 `raw/`에 넣습니다. `2018-04-03_varcharge`는 분석에 사용하지 않습니다.

원본 MAT와 중간 데이터·학습 모델은 Git에서 제외합니다. 중간 데이터는 `.cache/`에, 그래프와 성능표는 `results/`에 저장됩니다.

```bash
uv run python -m src.preprocess --rebuild
uv run python -m src.features
uv run python -m src.train
```
