# 원본 데이터

`raw/`에 MIT–Stanford Battery Dataset의 MAT 파일을 보관한다.

| 파일 | 사용 구분 |
|---|---|
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | Batch 1 · 학습 |
| 2018-02-20_batchdata_updated_struct_errorcorrect.mat | Batch 2 · 평가 |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | Batch 3 · 추가 평가 |
| 2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat | 별도 실험 · 보관 |

원본 출처: [MIT–Stanford Battery Dataset](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204).
원본 MAT는 용량이 커 Git에서 제외한다. 실행에 필요한 것은 위의 Batch 1·2·3 파일이며 extra 파일은 분석에 사용하지 않는다. 제공받은 MAT는 파일명을 변경하지 않고 raw/에 넣는다.

EDA 노트북은 원본 MAT를 직접 읽는다. 피처 생성은 같은 원본에서 추출한 중간 데이터를 사용하며 원본 변경 시 다시 추출한다. 중간 데이터·피처·학습 모델은 Git에서 제외되는 `.cache`에 자동 생성한다. `results`에는 `figures/`와 `model_performance.csv`만 저장한다.

```bash
uv run python -m src.preprocess --rebuild
uv run python -m src.features
uv run python -m src.train
```
