"""피처 생성부터 후보 비교, 최종 학습, 외부 배치 평가까지 실행한다.

1. B1을 개발 26셀 / 정책 Hold-out 10셀로 나눈다.
2. 개발군에서 11개 모델 × 5개 피처군을 nested CV로 비교한다.
3. 개발 CV로 후보를 고정한 뒤 Hold-out을 평가한다.
4. 선택 모델의 B1 전체 CV를 보고하고, B1 전체로 재학습한다.
5. 동일 모델을 B2 / B3에 적용한다. 테스트 점수로 다시 선택하지 않는다.

중간 실험 기록은 .cache/modeling, 제출 성능표는 results에 저장한다."""
from pathlib import Path
import os
import json
import hashlib
import warnings
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, make_scorer
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
OUT = ROOT / '.cache/modeling'
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
SEED = int(json.loads((ROOT / 'config.json').read_text())['seed'])


def experiment_fingerprint():
    """현재 원본·설정·실행 코드를 기록해 오래된 학습 결과를 구분한다."""
    from src.preprocess import source_fingerprint

    dataset = Path(os.environ.get('BATTERY_DATA_DIR', str(ROOT / 'data/raw')))
    inputs = [ROOT / 'config.json', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    inputs.extend(sorted((ROOT / 'src').rglob('*.py')))
    return {
        'raw': source_fingerprint(dataset),
        'files': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
    }


def results_are_current():
    """끝까지 완료된 실험이며 입력과 결과 파일이 그대로인 경우에만 재사용한다."""
    manifest = OUT / 'completed_run.json'
    if not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text())
        if record['inputs'] != experiment_fingerprint():
            return False
        for name, expected_hash in record['outputs'].items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                return False
        return True
    except (KeyError, json.JSONDecodeError):
        return False


def mape_log(a, p):
    """log10 타깃을 실제 cycles로 역변환한 뒤 MAPE(%)를 계산한다."""
    return 100 * mean_absolute_percentage_error(10 ** a, 10 ** p)
# GridSearchCV는 점수가 클수록 좋다고 본다. MAPE는 작을수록 좋아 음수 점수로 전달한다.
SCORER = make_scorer(mape_log, greater_is_better=False)


def metric(a, p):
    """실제 cycles 기준 MAPE·MAE·RMSE·R²·평균 잔차를 반환한다."""
    # MAPE는 %, MAE·RMSE·평균 잔차는 cycles 단위다. 양수 잔차는 수명을 길게 예측했다는 뜻이다.
    return {
        'n': len(a),
        'MAPE_pct': float(100 * mean_absolute_percentage_error(a, p)),
        'MAE_cycles': float(mean_absolute_error(a, p)),
        'RMSE_cycles': float(np.sqrt(mean_squared_error(a, p))),
        'R2': float(r2_score(a, p)),
        'bias_cycles': float(np.mean(np.asarray(p) - np.asarray(a))),
    }


def pipeline(model):
    """각 fold에서 새로 적합할 대치 → 표준화 → 모델 Pipeline을 만든다."""
    # 검증 셀의 정보로 중앙값이나 평균을 구하지 않도록 전처리도 모델과 함께 fit한다.
    return Pipeline(
        [
            ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('scale', StandardScaler()),
            ('model', clone(model)),
        ],
    )


def specs():
    """후보 11개 모델의 고정 설정과 내부 CV에서 탐색할 파라미터를 반환한다."""
    return {
        'linear': (LinearRegression(), {}),
        'ridge': (Ridge(), {'model__alpha': [0.01, 0.1, 1, 10, 100]}),
        'elastic_net': (ElasticNet(max_iter=50000, tol=1e-05), {'model__alpha': [0.001, 0.01, 0.1], 'model__l1_ratio': [0.2, 0.8]}),
        'huber': (HuberRegressor(max_iter=5000), {'model__alpha': [0.01], 'model__epsilon': [1.35, 1.8]}),
        'bayesian_ridge': (BayesianRidge(), {}),
        'svr': (SVR(), {'model__C': [1, 10], 'model__epsilon': [0.02]}),
        'random_forest': (RandomForestRegressor(n_estimators=120, min_samples_leaf=3, n_jobs=1, random_state=SEED), {'model__max_depth': [2, 4]}),
        'extra_trees': (ExtraTreesRegressor(n_estimators=120, min_samples_leaf=3, n_jobs=1, random_state=SEED), {'model__max_depth': [2, 4]}),
        'xgboost': (XGBRegressor(
            n_estimators=150,
            learning_rate=0.035,
            min_child_weight=3,
            reg_lambda=5,
            n_jobs=1,
            tree_method='hist',
            random_state=SEED,
        ), {'model__max_depth': [1, 2]}),
        'lightgbm': (LGBMRegressor(
            n_estimators=150,
            learning_rate=0.035,
            min_child_samples=4,
            reg_lambda=5,
            n_jobs=1,
            verbosity=-1,
            random_state=SEED,
        ), {'model__num_leaves': [3, 5]}),
        'catboost': (CatBoostRegressor(
            iterations=150,
            learning_rate=0.035,
            l2_leaf_reg=5,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            random_seed=SEED,
        ), {'model__depth': [2, 3]}),
    }


def main():
    """B1 내부 선택부터 B2/B3 평가까지 실행하고 결과 파일을 저장한다."""
    # 재실행이 중간에 실패하면 이전 실험을 이번 실행의 완료 결과로 읽지 않게 한다.
    (OUT / 'completed_run.json').unlink(missing_ok=True)
    input_fingerprint = experiment_fingerprint()
    config = json.loads((ROOT / 'config.json').read_text())
    from src.features import build_features
    feature_table, _ = build_features()
    paper = {
        'feature_families': {
            'discharge_selected6': ['dq_logmin', 'dq_logvar', 'dq_logskew', 'dq_logkurtosis', 'q2', 'q_max_minus_q2'],
        },
    }
    families = {
        'dq_only': ['dq_logvar'],
        'capacity': config['features']['ablations']['capacity'],
        'capacity_ir_temperature': config['features']['ablations']['capacity_ir_temperature'],
        'paper_discharge6': paper['feature_families']['discharge_selected6'],
        'core6': config['features']['core'],
    }
    assert not set(sum(families.values(), [])) & set(config['features']['forbidden_inputs'])
    # 1. 선택은 B1 안에서만 한다. 같은 숫자 충전 프로토콜의 셀은 같은 그룹에 둔다.
    batch1 = feature_table[feature_table.batch == 'B1'].reset_index(drop=True)
    # test_size=0.25는 셀 수가 아니라 프로토콜 그룹의 비율이다. 실제 분리는 개발 26셀 / 검증 10셀이다.
    development_indices, holdout_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(
            batch1,
            groups=batch1.policy_group,
        ),
    )
    development = batch1.iloc[development_indices].reset_index(drop=True)
    holdout = batch1.iloc[holdout_indices].reset_index(drop=True)
    assert set(development.policy_group).isdisjoint(holdout.policy_group)
    # 실험 조건과 분리 목록을 남겨 나중에 어떤 데이터로 선택했는지 확인할 수 있게 한다.
    plan = {
        'seed': SEED,
        'cohort_counts': feature_table.batch.value_counts().to_dict(),
        'selection_batch': 'B1 only',
        'holdout': {
            'development_cells': development.cell.tolist(),
            'validation_cells': holdout.cell.tolist(),
            'test_size_policy_groups': 0.25,
        },
        'families': families,
        'models': {k: {'class': type(v[0]).__name__, 'fixed_params': v[0].get_params(), 'grid': v[1]} for k, v in specs().items()},
        'selection': 'lowest mean development nested4fold GroupKFold MAPE, then lower feature count tie-break',
        'nested': 'outer4 policy folds on development26, inner3 policy folds tuning; selected family fixed in each candidate',
        'Train_reporting': 'selected family/model nested5fold GroupKFold on all36 B1 cells, inner3fold tuning, unweighted outer-fold MAPE mean',
        'Valid_reporting': 'selected model tuned/refit development26; predict untouched10 holdout before any fullB1 refit',
        'Test_reporting': 'reuse development26-selected hyperparameters, refit fullB1 36 then evaluate B2/B3, no retuning',
        'target': 'log10 life, inverse10**prediction, scoring MAPE in cycles',
        'gap_sign': 'Train-Valid label computes Valid minus Train; Valid-Test computes Test minus Valid; Target-Test computes Test minus9.1; Batch2-Batch3 computes B3 minus B2, all percentage points',
        'limitations': [
            'All batches previously inspected in EDA and exploratory sweeps; this is compliance-oriented exploratory evaluation, not a newly blind test',
            'Hyperparameter grids bounded; best means best within these candidates',
            'Group holdout prevents shared protocol; plain cell-only holdout does not guarantee this',
        ],
    }
    (OUT / 'frozen_plan.json').write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, default=str) + '\n',
    )
    warning_records = []
    failures = []
    candidates = []
    development_fold_scores = []
    development_predictions = []

    def tune(model, grid, train):
        """학습 데이터의 정책을 분리한 내부 3-fold GridSearchCV를 만든다. 여기서는 아직 fit하지 않는다."""
        # 내부 3-fold는 파라미터 선택용이다. 그 바깥의 fold가 선택 결과의 오차를 평가한다.
        cv = list(GroupKFold(3).split(train, groups=train.policy_group))
        for a, b in cv:
            assert set(train.policy_group.iloc[a]).isdisjoint(train.policy_group.iloc[b])
        search = GridSearchCV(pipeline(model), grid, cv=cv, scoring=SCORER, n_jobs=1, error_score='raise')
        return search
    with threadpool_limits(limits=1):
        # 2. 개발군의 외부 4-fold로 11개 모델 × 5개 피처군을 비교한다. Hold-out은 아직 쓰지 않는다.
        outer = list(GroupKFold(4).split(development, groups=development.policy_group))
        for name, (model, grid) in specs().items():
            for family, feature_names in families.items():
                scores = []
                rows = []
                try:
                    for fold, (a, b) in enumerate(outer):
                        train = development.iloc[a]
                        test = development.iloc[b]
                        assert set(train.policy_group).isdisjoint(test.policy_group)
                        search = tune(model, grid, train)
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter('always')
                            search.fit(train[feature_names], train.target_log10)
                        if ws:
                            warning_records.append(
                                {
                                    'stage': 'development',
                                    'model': name,
                                    'family': family,
                                    'fold': fold,
                                    'warnings': sorted(set((str(w.message) for w in ws))),
                                },
                            )
                        pred = 10 ** search.predict(test[feature_names])
                        assert np.isfinite(pred).all()
                        m = metric(test.life, pred)
                        scores.append(m['MAPE_pct'])
                        development_fold_scores.append({'model': name, 'family': family, 'fold': fold, **m})
                        for r, v in zip(test.itertuples(), pred):
                            rows.append(
                                {
                                    'model': name,
                                    'family': family,
                                    'fold': fold,
                                    'cell': r.cell,
                                    'actual': r.life,
                                    'predicted': v,
                                },
                            )
                    candidates.append(
                        {
                            'model': name,
                            'family': family,
                            'n_features': len(feature_names),
                            'development_CV_MAPE_mean': np.mean(scores),
                            'development_CV_MAPE_std': np.std(scores, ddof=1),
                        },
                    )
                    development_predictions += rows
                # 한 후보가 실패해도 나머지 비교는 진행하고, 실패 이유를 별도 기록한다.
                except Exception as e:
                    failures.append({'model': name, 'family': family, 'error': repr(e)})
            print('candidate completed', name, flush=True)
        ranking = pd.DataFrame(candidates).sort_values(['development_CV_MAPE_mean', 'n_features', 'model']).reset_index(
            drop=True,
        )
        ranking.to_csv(OUT / 'candidate_ranking.csv', index=False)
        pd.DataFrame(development_fold_scores).to_csv(OUT / 'development_fold_scores.csv', index=False)
        pd.DataFrame(development_predictions).to_csv(OUT / 'development_predictions.csv', index=False)
        # 개발 CV 평균 MAPE가 가장 작은 후보를 고른다. 동점이면 피처 수가 적은 후보가 앞선다.
        best = ranking.iloc[0]
        name = best.model
        family = best.family
        feature_names = families[family]
        model, grid = specs()[name]
        selection = {
            'model': name,
            'family': family,
            'features': feature_names,
            'criterion': 'development nested policy CV MAPE',
            'CV_MAPE': float(best.development_CV_MAPE_mean),
        }
        # 3. 후보를 고정한 뒤 개발군으로 학습하고, 분리해 둔 Hold-out 10셀을 평가한다.
        selected = tune(model, grid, development)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            selected.fit(development[feature_names], development.target_log10)
        if ws:
            warning_records.append(
                {'stage': 'selected_development_fit', 'warnings': sorted(set((str(w.message) for w in ws)))},
            )
        selection['parameters'] = selected.best_params_
        (OUT / 'selection_before_test.json').write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
        )
        joblib.dump(selected.best_estimator_, OUT / 'validation_pipeline.joblib')
        holdout_predictions = 10 ** selected.predict(holdout[feature_names])
        holdout_metrics = metric(holdout.life, holdout_predictions)
        print('selected', name, family, 'valid MAPE', holdout_metrics['MAPE_pct'], flush=True)
        # 4. 보고용 B1 전체 5-fold CV다. 이미 선택한 모델·피처군을 쓰므로 완전히 독립적인 선택 성능은 아니다.
        batch1_fold_scores = []
        batch1_cv_predictions = []
        for fold, (a, b) in enumerate(GroupKFold(5).split(batch1, groups=batch1.policy_group)):
            train = batch1.iloc[a]
            test = batch1.iloc[b]
            search = tune(model, grid, train)
            assert set(train.policy_group).isdisjoint(test.policy_group)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                search.fit(train[feature_names], train.target_log10)
            if ws:
                warning_records.append(
                    {'stage': 'B1_report_CV', 'fold': fold, 'warnings': sorted(set((str(w.message) for w in ws)))},
                )
            pred = 10 ** search.predict(test[feature_names])
            batch1_fold_scores.append({'fold': fold, **metric(test.life, pred)})
            for r, v in zip(test.itertuples(), pred):
                batch1_cv_predictions.append(
                    {
                        'stage': 'B1_CV',
                        'fold': fold,
                        'cell': r.cell,
                        'batch': r.batch,
                        'actual': r.life,
                        'predicted': v,
                    },
                )
        pd.DataFrame(batch1_fold_scores).to_csv(OUT / 'b1_cv_fold_scores.csv', index=False)
        # 외부 배치용 모델은 개발군에서 정한 파라미터를 그대로 쓰고, B1 전체 36셀로 재학습한다.
        final = clone(selected.best_estimator_).fit(batch1[feature_names], batch1.target_log10)
        joblib.dump(final, OUT / 'final_pipeline.joblib')
        external = {}
        prediction_rows = batch1_cv_predictions.copy()
        for r, v in zip(holdout.itertuples(), holdout_predictions):
            prediction_rows.append(
                {'stage': 'Valid', 'cell': r.cell, 'batch': r.batch, 'actual': r.life, 'predicted': v},
            )
        # 5. 같은 확정 모델로 B2·B3을 평가한다. 이 점수를 보고 파라미터를 다시 고르지 않는다.
        for batch in ['B2', 'B3']:
            test = feature_table[feature_table.batch == batch]
            pred = 10 ** final.predict(test[feature_names])
            assert np.isfinite(pred).all() and (pred > 0).all()
            assert np.allclose(
                joblib.load(OUT / 'final_pipeline.joblib').predict(test[feature_names]),
                final.predict(test[feature_names]),
            )
            external[batch] = metric(test.life, pred)
            for r, v in zip(test.itertuples(), pred):
                prediction_rows.append(
                    {'stage': 'Test', 'cell': r.cell, 'batch': r.batch, 'actual': r.life, 'predicted': v},
                )
        predictions = pd.DataFrame(prediction_rows)
        predictions['APE_pct'] = abs(predictions.predicted / predictions.actual - 1) * 100
        predictions.to_csv(OUT / 'predictions.csv', index=False)
        # Train은 fold별 MAPE의 단순 평균이다. 전체 CV 예측을 합친 MAPE와는 구분한다.
        train_mape = float(np.mean([r['MAPE_pct'] for r in batch1_fold_scores]))
        batch2_mape = external['B2']['MAPE_pct']
        batch3_mape = external['B3']['MAPE_pct']
        # Gap은 오류 증가량(pp)이다. 표의 Train-Valid는 Valid−Train, Valid-Test는 Test−Valid로 계산한다.
        reporting = [
            {
                'index': 'Train (Batch 1 CV)',
                'MAPE_pct': train_mape,
                'note': 'B1 36셀 · 프로토콜 분리 nested5fold MAPE 평균',
            },
            {
                'index': 'Valid (Batch 1 Hold-out)',
                'MAPE_pct': holdout_metrics['MAPE_pct'],
                'note': '개발26셀로 학습 · 독립 프로토콜10셀 검증',
            },
            {'index': 'Test (Batch 2)', 'MAPE_pct': batch2_mape, 'note': 'B1 36셀 재학습 후 B2 39셀 평가'},
            {
                'index': 'Gap (Train-Valid)',
                'MAPE_pct': holdout_metrics['MAPE_pct'] - train_mape,
                'note': 'Valid−Train · pp · (+) 과적합 의심',
            },
            {
                'index': 'Gap (Valid-Test)',
                'MAPE_pct': batch2_mape - holdout_metrics['MAPE_pct'],
                'note': 'B2−Valid · pp · (+) 배치 일반화 저하',
            },
            {'index': 'Gap (Target-Test)', 'MAPE_pct': batch2_mape - 9.1, 'note': 'B2−9.1% · pp'},
            {'index': 'Test (Batch 3)', 'MAPE_pct': batch3_mape, 'note': '같은 확정 모델 · B3 43셀 평가'},
            {
                'index': 'Gap (Batch2-Batch3)',
                'MAPE_pct': batch3_mape - batch2_mape,
                'note': 'B3−B2 · pp · (+) B3 성능 저하',
            },
            {
                'index': 'Gap (Target-Test) — Batch 3',
                'MAPE_pct': batch3_mape - 9.1,
                'note': 'B3−9.1% · pp',
            },
        ]
        pd.DataFrame(reporting).to_csv(OUT / 'performance_reporting.csv', index=False)
        pd.DataFrame(reporting).to_csv(RESULTS / 'model_performance.csv', index=False)
        result = {
            'selected': selection,
            'Train_CV_MAPE_mean': train_mape,
            'Train_CV_MAPE_std': float(np.std([r['MAPE_pct'] for r in batch1_fold_scores], ddof=1)),
            'Train_CV_pooled': metric(pd.DataFrame(batch1_cv_predictions).actual, pd.DataFrame(batch1_cv_predictions).predicted),
            'Valid': holdout_metrics,
            'Test': external,
            'reporting': reporting,
        }
        (OUT / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (OUT / 'warnings.json').write_text(json.dumps(warning_records, ensure_ascii=False, indent=2))
        (OUT / 'failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2))
        roles = []
        for r in feature_table.itertuples():
            roles.append(
                {
                    'cell': r.cell,
                    'batch': r.batch,
                    'policy_group': r.policy_group,
                    'role': ('development' if r.cell in set(development.cell) else 'holdout') if r.batch == 'B1' else 'external_test',
                    'life': r.life,
                },
            )
        pd.DataFrame(roles).to_csv(OUT / 'split_assignments.csv', index=False)
        (OUT / 'verification.json').write_text(
            json.dumps(
                {
                    'B1_only_selection': True,
                    'development_n': 26,
                    'holdout_n': 10,
                    'policy_overlap_dev_valid': 0,
                    'B1_CV_unique_cells': len(pd.DataFrame(batch1_cv_predictions).cell.unique()),
                    'B2_n': 39,
                    'B3_n': 43,
                    'final_model_fit_cells': batch1.cell.tolist(),
                    'final_reload_predictions_match': True,
                    'candidate_count': len(ranking),
                    'failed_candidates': len(failures),
                    'warning_groups': len(warning_records),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        # 보고서 파일은 학습 입력이 아니다. 보고서가 없어도 학습·평가는 완료된다.
        # 03에서 읽는 표와 확정 모델까지 저장된 뒤에만 완료 기록을 남긴다.
        completed_outputs = [
            OUT / 'metrics.json', OUT / 'candidate_ranking.csv',
            OUT / 'b1_cv_fold_scores.csv', OUT / 'predictions.csv',
            OUT / 'final_pipeline.joblib', OUT / 'validation_pipeline.joblib',
            OUT / 'selection_before_test.json', OUT / 'split_assignments.csv',
            ROOT / '.cache/features/engineered_features.csv',
            RESULTS / 'model_performance.csv',
        ]
        (OUT / 'completed_run.json').write_text(json.dumps({
            'inputs': input_fingerprint,
            'outputs': {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in completed_outputs
            },
        }, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
