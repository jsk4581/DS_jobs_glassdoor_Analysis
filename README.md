# DS_jobs_glassdoor_Analysis
DataScience Team 9
https://www.kaggle.com/datasets/fahadrehman07/data-science-jobs-and-salary-glassdoor/



# Glassdoor Data Science Project Documentation

## Table of Contents

- [Glassdoor Data Science Project Documentation](#glassdoor-data-science-project-documentation)
  - [Table of Contents](#table-of-contents)
- [Data Processing Methods](#data-processing-methods)
    - [`extract_salary(s: str) -> Tuple[float, float, float]`](#extract_salarys-str---tuplefloat-float-float)
    - [`map_size(size_str: str) -> str`](#map_sizesize_str-str---str)
    - [`df['Company_age'] = 2025 - df['Founded']`](#dfcompany_age--2025---dffounded)
    - [`revenue_to_numeric(revenue: str) -> float`](#revenue_to_numericrevenue-str---float)
    - [`map_state_to_region(state: str) -> str`](#map_state_to_regionstate-str---str)
    - [`process_location_columns(df: pd.DataFrame) -> pd.DataFrame`](#process_location_columnsdf-pddataframe---pddataframe)
    - [`classify_seniority(title: str) -> str`](#classify_senioritytitle-str---str)
    - [`classify_ownership(value: str) -> str`](#classify_ownershipvalue-str---str)
    - [결측치 처리 (Missing Value Imputation)](#결측치-처리-missing-value-imputation)
    - [최종 피처 정리 및 저장](#최종-피처-정리-및-저장)
- [Recommendation\_System\_Clustering](#recommendation_system_clustering)
  - [`preprocess_salary_and_size(df: pd.DataFrame) -> pd.DataFrame`](#preprocess_salary_and_sizedf-pddataframe---pddataframe)
  - [`drop_remaining_nas(df: pd.DataFrame) -> pd.DataFrame`](#drop_remaining_nasdf-pddataframe---pddataframe)
  - [`prepare_features_for_clustering(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]`](#prepare_features_for_clusteringdf-pddataframe-features-liststr---tuplenpndarray-standardscaler-pddataframe)
  - [`plot_elbow_method(X: np.ndarray, k_range=range(2, 7)) -> None`](#plot_elbow_methodx-npndarray-k_rangerange2-7---none)
  - [`run_kmeans(X: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]`](#run_kmeansx-npndarray-k-int---tuplekmeans-npndarray)
  - [`visualize_clusters(X: np.ndarray, labels: np.ndarray) -> None`](#visualize_clustersx-npndarray-labels-npndarray---none)
  - [`recommend_similar_jobs(user_input: dict, df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, df_encoded_cat: pd.DataFrame) -> pd.DataFrame`](#recommend_similar_jobsuser_input-dict-df-pddataframe-kmeans-kmeans-scaler-standardscaler-df_encoded_cat-pddataframe---pddataframe)
- [Salary\_Imputation\_Regression](#salary_imputation_regression)
  - [`evaluate(pipe: Pipeline) -> dict`](#evaluatepipe-pipeline---dict)
  - [`get_feature_names(pipe: Pipeline) -> np.ndarray`](#get_feature_namespipe-pipeline---npndarray)
  - [`evaluate_and_visualize_models()`](#evaluate_and_visualize_models)
  - [`predict_missing_avg_salary(best_pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame`](#predict_missing_avg_salarybest_pipeline-pipeline-df-pddataframe---pddataframe)
  - [`save_final_modeling_output(df: pd.DataFrame, file_name: str) -> None`](#save_final_modeling_outputdf-pddataframe-file_name-str---none)
- [Salary\_Prediction\_Regression](#salary_prediction_regression)
  - [`preprocess_features(X: pd.DataFrame) -> np.ndarray`](#preprocess_featuresx-pddataframe---npndarray)
  - [`evaluate_model(model, X, y) -> Tuple[float, float, float, float]`](#evaluate_modelmodel-x-y---tuplefloat-float-float-float)
  - [Random Forest 하이퍼파라미터 튜닝 루프](#random-forest-하이퍼파라미터-튜닝-루프)
  - [XGBoost 하이퍼파라미터 튜닝 루프](#xgboost-하이퍼파라미터-튜닝-루프)
  - [`results_df` 생성 및 최적 조합 추출](#results_df-생성-및-최적-조합-추출)
  - [최종 모델 학습 및 시각화](#최종-모델-학습-및-시각화)
  - [`plot_feature_importance(model, feature_names: list, top_n=15) -> None`](#plot_feature_importancemodel-feature_names-list-top_n15---none)
- [Salary\_Modeling\_Analysis](#salary_modeling_analysis)
  - [결측치 처리 루틴](#결측치-처리-루틴)
  - [`evaluate_model(model, X, y) -> Tuple[float, float, float, float]`](#evaluate_modelmodel-x-y---tuplefloat-float-float-float-1)
  - [하이퍼파라미터 튜닝 (GridSearchCV / RandomizedSearchCV)](#하이퍼파라미터-튜닝-gridsearchcv--randomizedsearchcv)
    - [Random Forest (`GridSearchCV`)](#random-forest-gridsearchcv)
    - [XGBoost (`GridSearchCV`)](#xgboost-gridsearchcv)
  - [로그 변환 및 역변환 성능 평가](#로그-변환-및-역변환-성능-평가)
  - [이상치 제거 후 재학습](#이상치-제거-후-재학습)
  - [Feature Importance 시각화](#feature-importance-시각화)
- [Company\_Rating\_Prediction](#company_rating_prediction)
  - [Processing](#processing)
    - [`extract_salary(s: str) → Tuple[float, float, float]`](#extract_salarys-str--tuplefloat-float-float)
    - [`map_size(size_str: str) → str`](#map_sizesize_str-str--str)
    - [`classify_domain(title: str) → str`](#classify_domaintitle-str--str)
    - [`classify_seniority(title: str) → str`](#classify_senioritytitle-str--str)
    - [`fill_features(row: pd.Series) → pd.Series`](#fill_featuresrow-pdseries--pdseries)
    - [`classify_ownership(val: str) → str`](#classify_ownershipval-str--str)
  - [Modeling](#modeling)
    - [`prepare_data(df: pd.DataFrame) → Tuple[X, y_reg, y_cls]`](#prepare_datadf-pddataframe--tuplex-y_reg-y_cls)
    - [`evaluate_linear_regression(X, y_reg, test_sizes) → List[Tuple[float, float, float]]`](#evaluate_linear_regressionx-y_reg-test_sizes--listtuplefloat-float-float)
    - [`evaluate_decision_tree(X, y_cls, max_depths) → List[Tuple[int, float]]`](#evaluate_decision_treex-y_cls-max_depths--listtupleint-float)
    - [`final_model_evaluation_and_selection() → None`](#final_model_evaluation_and_selection--none)
    - [`plot_cross_validation_results(cv_scores: np.ndarray) → None`](#plot_cross_validation_resultscv_scores-npndarray--none)


# Data Processing Methods

### `extract_salary(s: str) -> Tuple[float, float, float]`
- **목적**: Salary Estimate 문자열을 파싱하여 최소, 최대, 평균 연봉을 숫자로 추출
- **이유**: 연봉 정보는 핵심 타깃 변수이며, 수치형 변환을 통해 분석 및 예측에 활용할 수 있도록 함
- **입력**: `s` (연봉 범위를 포함하는 문자열 예: `"$70K-$90K (Glassdoor est.)"`)
- **출력**: `min_salary`, `max_salary`, `avg_salary` (단위: 천 달러)

### `map_size(size_str: str) -> str`
- **목적**: Size 컬럼을 범주형 그룹(예: Small, Medium, Large)으로 매핑
- **이유**: 범위가 다양한 기업 규모를 간단한 그룹으로 정리하여 분석 편의성 향상
- **입력**: `size_str` (기업 규모 문자열)
- **출력**: 그룹화된 범주 문자열 (예: `'Medium'`)

### `df['Company_age'] = 2025 - df['Founded']`
- **목적**: 설립년도(Founded)를 활용하여 회사의 나이를 계산
- **이유**: 설립연도보다 "운영 기간"이 더 직관적인 수치형 변수로 활용 가능
- **출력**: `Company_age` (2025년 기준)

### `revenue_to_numeric(revenue: str) -> float`
- **목적**: Revenue 문자열을 정규화된 수치로 변환
- **이유**: 문자열 매출 범위는 모델 학습에 사용될 수 없으므로 수치화 필요
- **입력**: `revenue` (예: `"$5 to $10 billion"`)
- **출력**: 평균 매출 값 (단위: 달러, 백만~십억 단위)

### `map_state_to_region(state: str) -> str`
- **목적**: 회사 위치(Location, Headquarters)의 주(State)를 기반으로 지역(Region) 파생
- **이유**: 지역 단위 분석을 통해 고용/급여 패턴 차이 분석 가능
- **입력**: `state` (주 문자열)
- **출력**: 지역명 (예: `'South'`, `'West'`)

### `process_location_columns(df: pd.DataFrame) -> pd.DataFrame`
- **목적**: Location과 Headquarters 컬럼에서 주(State) 및 지역(Region)을 파생하며, 결측값은 최빈값으로 보완
- **이유**: 기업 위치는 연봉, 산업군, 기업 특성 등과 밀접하게 연결되어 있으며, 주(State) 및 지역(Region) 단위로 파생해 분석 정확도를 높임
- **주요 처리 단계**:
  - Location, Headquarters 컬럼 결측값을 최빈값으로 채움
  - 각 컬럼에서 마지막 두 글자를 슬라이싱하여 주(State) 추출
  - 주(State)를 기반으로 사전에 정의된 딕셔너리로 지역(Region) 파생
  - 추출된 State/Region 컬럼을 추가
- **출력**: `Location_state`, `Location_region`, `Headquarters_state`, `Headquarters_region` 컬럼이 포함된 DataFrame

### `classify_seniority(title: str) -> str`
- **목적**: Job Title에서 직급을 분류
- **이유**: 역할/수준별 급여 예측 정확도를 향상시키기 위한 파생 피처 생성
- **출력 예시**: `'Senior'`, `'Junior'`

### `classify_ownership(value: str) -> str`
- **목적**: Ownership Type을 `'Private'`, `'Public'`, `'Public Service'`로 정리
- **이유**: 분석 가능한 적은 범주의 소유형으로 단순화

### 결측치 처리 (Missing Value Imputation)
- **목적**: 결측값이 있는 피처에 대해 적절한 방식으로 보완
- **이유**: 머신러닝 알고리즘은 결측치를 허용하지 않기 때문에 필수 작업
- **전략**:
  - 수치형 피처: 그룹 평균 또는 전체 평균으로 채움
  - 범주형 피처: 최빈값으로 채움
  - `Rating == -1`: 전체 평균으로 대체

### 최종 피처 정리 및 저장
- **목적**: 분석 및 학습에 사용할 피처만 정리하여 `.csv` 파일로 저장
- **이유**: 노이즈 피처를 제거하고 파이프라인 학습 효율을 높이기 위함
- **출력 파일**:
  - `glassdoor_cleaned.csv`: 전처리된 주요 피처 포함
  - `glassdoor_cleaned_final.csv`: 결측치 처리 및 최종 완성본


# Recommendation_System_Clustering

##  `preprocess_salary_and_size(df: pd.DataFrame) -> pd.DataFrame`

- **목적**: 평균, 최소, 최대 연봉 결측치를 직무, 직급, 산업군 그룹별 중앙값으로 채움. `Size_cleaned` 열을 오디널 인코딩하여 수치형 변수로 변환  
- **이유**: KMeans는 결측치를 허용하지 않으며, 범주형 변수는 수치형으로 변환되어야 함  
- **입력**:  
  - `df`: Glassdoor 원본 데이터프레임  
- **출력**:  
  - 일부 결측치가 채워진 전처리된 `DataFrame`

```python
df['avg_salary'] = df.groupby(['Job Title', 'Position_Encoded', 'Sector'])['avg_salary'].transform(lambda x: x.fillna(x.median()))
df['min_salary'] = df.groupby(['Job Title', 'Position_Encoded', 'Sector'])['min_salary'].transform(lambda x: x.fillna(x.median()))
df['max_salary'] = df.groupby(['Job Title', 'Position_Encoded', 'Sector'])['max_salary'].transform(lambda x: x.fillna(x.median()))

size_map = {'Small': 1, 'Medium': 2, 'Large': 3, 'Very Large': 4}
df['Size_cleaned'] = df['Size_cleaned'].map(size_map)
```

## `drop_remaining_nas(df: pd.DataFrame) -> pd.DataFrame`

- **목적**: 전처리 이후에도 남은 결측치가 있는 행 제거  
- **이유**: KMeans는 결측값이 있는 데이터를 처리하지 못함  
- **입력**:  
  - 전처리된 `DataFrame`  
- **출력**:  
  - 결측치가 제거된 `DataFrame`

```python
df_cleaned = df.dropna()
```

## `prepare_features_for_clustering(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]`

- **목적**: 수치형 변수는 표준화하고, 범주형 변수는 One-hot 인코딩하여 클러스터링용 피처 행렬 생성  
- **이유**: KMeans는 거리 기반 알고리즘이므로 모든 피처가 수치형이어야 하고 스케일에 민감함  
- **입력**:  
  - `df`: 결측치 제거된 데이터프레임  
  - `features`: 사용할 피처 리스트  
- **출력**:  
  - 정규화된 피처 행렬 `X_final`  
  - 학습된 `StandardScaler` 객체  
  - 인코딩된 범주형 피처 템플릿  

```python
df_encoded_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=df.index)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols + ord_cols])
X_final = np.hstack([X_scaled, df_encoded_cat.values])
```

## `plot_elbow_method(X: np.ndarray, k_range=range(2, 7)) -> None`

- **목적**: Elbow Method를 사용하여 최적 클러스터 수 시각적으로 탐색  
- **이유**: 적절한 클러스터 수(k)는 모델 성능에 중요  
- **입력**:  
  - 클러스터링용 피처 행렬 `X`  
  - 클러스터 수 후보 범위 `k_range`  
- **출력**:  
  - Elbow 그래프  

```python
inertias = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()
```

## `run_kmeans(X: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]`

- **목적**: KMeans 알고리즘으로 클러스터링 수행  
- **이유**: 유사 기업/직무를 클러스터로 묶어 분석 및 추천에 활용  
- **입력**:  
  - `X`: 정규화된 피처 행렬  
  - `k`: 클러스터 수  
- **출력**:  
  - 학습된 `KMeans` 객체  
  - 클러스터 라벨 벡터  

```python
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
```

## `visualize_clusters(X: np.ndarray, labels: np.ndarray) -> None`

- **목적**: PCA를 통해 고차원 데이터를 2D로 축소 후 시각화  
- **이유**: 클러스터링 결과를 직관적으로 이해하기 위해 시각화가 필요  
- **입력**:  
  - `X`: 정규화된 피처 행렬  
  - `labels`: 클러스터 라벨  
- **출력**:  
  - 2차원 산점도 그래프  

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('Cluster Visualization (PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
```

## `recommend_similar_jobs(user_input: dict, df: pd.DataFrame, kmeans: KMeans, scaler: StandardScaler, df_encoded_cat: pd.DataFrame) -> pd.DataFrame`

- **목적**: 사용자 입력 기반으로 가장 가까운 클러스터를 찾아 유사 공고 10개 추천  
- **이유**: 클러스터 내에서 가장 유사한 직무를 맞춤 추천  
- **입력**:  
  - `user_input`: 사용자 입력 사전 (예: `{'avg_salary': 100, 'Sector': 'Tech', ...}`)  
  - `df`: 클러스터 결과 포함된 전체 데이터프레임  
  - `kmeans`: 학습된 KMeans 모델  
  - `scaler`: 피처 스케일러  
  - `df_encoded_cat`: 인코딩된 범주형 피처 템플릿  
- **출력**:  
  - 추천 공고 상위 10개를 포함한 `DataFrame`

```python
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df[num_cols + ord_cols])
input_encoded = pd.get_dummies(input_df[cat_cols])
input_encoded = input_encoded.reindex(columns=df_encoded_cat.columns, fill_value=0)
input_final = np.hstack([input_scaled, input_encoded.values])

nearest_cluster = np.argmin(np.sum((kmeans.cluster_centers_ - input_final)**2, axis=1) ** 0.5)
recommendations = df[df['cluster'] == nearest_cluster].copy()

recommendations['similarity'] = np.linalg.norm(X[df['cluster'] == nearest_cluster] - input_final, axis=1)
top10 = recommendations.sort_values('similarity').head(10)
```



# Salary_Imputation_Regression

## `evaluate(pipe: Pipeline) -> dict`

- 목적: 주어진 회귀 모델 파이프라인에 대해 10-fold 2회 반복 교차검증을 수행하고, 주요 회귀 성능 지표(R², RMSE, MAE)의 평균값을 계산  
- 이유: 다양한 모델을 객관적 지표로 비교하여 최적 모델을 선택하기 위함  
- 입력:  
  - `pipe`: 사전 정의된 Pipeline 객체 (전처리 + 회귀 모델 포함)  
- 출력:  
  - 성능 지표 딕셔너리 (예: `{'R2_mean': ..., 'RMSE_mean': ..., 'MAE_mean': ...}`)

```python
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=42)

r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='r2')
rmse = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=make_scorer(mean_squared_error, squared=False))
mae = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')

result = {
    'R2_mean': np.mean(r2),
    'RMSE_mean': np.mean(rmse),
    'MAE_mean': -np.mean(mae)
}
```

## `get_feature_names(pipe: Pipeline) -> np.ndarray`

- 목적: ColumnTransformer와 OneHotEncoder 이후의 최종 피처 이름 배열을 반환  
- 이유: 회귀 계수(coef_)나 feature importance 시각화를 위해 정확한 피처 이름이 필요  
- 입력:  
  - 학습이 완료된 `Pipeline` 객체  
- 출력:  
  - 모든 수치형 + 인코딩된 범주형 피처 이름 배열 (`np.ndarray`)

```python
ohe = pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_names = ohe.get_feature_names_out(categorical_features)

feature_names = np.concatenate([numerical_features, cat_names])
```

## `evaluate_and_visualize_models()`

- 목적: 세 가지 회귀 모델 (Linear Regression, Random Forest, Gradient Boosting)을 학습 및 평가  
  - 모델별 성능 지표를 비교표로 생성  
  - 모델별 중요 피처를 시각화  
- 이유: 예측력이 가장 우수한 모델 선택 + 주요 영향 피처를 통해 인사이트 확보  
- 입력:  
  - 사전 정의된 데이터셋 (`X_train`, `y_train`), 파이프라인  
- 출력:  
  - `results_df`: 평가 결과 DataFrame  
  - `best_pipeline`: 가장 성능이 우수한 모델  
  - 피처 중요도 바 그래프 (matplotlib)

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import pandas as pd

models = {
    'Linear': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

results = []
for name, model in models.items():
    pipe = Pipeline([...])  # 전처리 포함
    score = evaluate(pipe)
    results.append({'Model': name, **score})

results_df = pd.DataFrame(results)
best_model_name = results_df.sort_values('R2_mean', ascending=False).iloc[0]['Model']
best_pipeline = models[best_model_name]

# 피처 중요도 시각화 (RandomForest, GradientBoosting일 때만)
if hasattr(best_pipeline, 'feature_importances_'):
    importances = best_pipeline.feature_importances_
    plt.barh(feature_names, importances)
    plt.title("Feature Importances")
    plt.show()
```


## `predict_missing_avg_salary(best_pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame`

- 목적: 학습된 최적 모델을 활용하여 결측된 `avg_salary` 값을 예측하여 보완  
- 이유: 타깃 변수의 결측치를 제거하지 않고 활용하여 데이터 손실 최소화  
- 입력:  
  - `best_pipeline`: 가장 우수한 회귀 파이프라인  
  - `df`: 결측값이 포함된 원본 DataFrame  
- 출력:  
  - `avg_salary` 결측치가 채워진 DataFrame

```python
X_missing = df[df['avg_salary'].isna()].drop(columns=['avg_salary'])
df.loc[df['avg_salary'].isna(), 'avg_salary'] = best_pipeline.predict(X_missing)
```


## `save_final_modeling_output(df: pd.DataFrame, file_name: str) -> None`

- 목적: `min_salary`, `max_salary` 컬럼을 제거하고, `avg_salary`를 마지막 컬럼으로 정렬하여 저장  
- 이유: 후속 분석 및 저장 포맷 정리에 유리한 형태로 출력  
- 입력:  
  - `df`: 회귀 결과를 포함한 DataFrame  
  - `file_name`: 저장할 `.csv` 파일명  
- 출력:  
  - 없음 (지정된 파일로 저장)

```python
df = df.drop(columns=['min_salary', 'max_salary'])
cols = [col for col in df.columns if col != 'avg_salary'] + ['avg_salary']
df = df[cols]
df.to_csv(file_name, index=False)
```


# Salary_Prediction_Regression

## `preprocess_features(X: pd.DataFrame) -> np.ndarray`

- 목적: 수치형 피처는 평균 대체 및 표준화, 범주형 피처는 최빈값 대체 및 One-hot 인코딩을 수행하여 모델 학습용 피처 행렬을 생성  
- 필요성: 결측치와 범주형 변수는 모델 학습에 적합하지 않기 때문에 전처리가 필수적임  
- 입력:  
  - `X`: 원본 피처 DataFrame (수치형 + 범주형)  
- 출력:  
  - `X_processed`: 전처리된 입력 데이터 (`np.ndarray`)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

X_processed = preprocessor.fit_transform(X)
```

## `evaluate_model(model, X, y) -> Tuple[float, float, float, float]`

- 목적: 주어진 회귀 모델에 대해 5-Fold 교차검증을 수행하고, 평균 MAE, MSE, RMSE, R² 값을 산출  
- 필요성: 모델 성능을 객관적이고 일관된 기준으로 비교할 수 있게 함  
- 입력:  
  - `model`: 회귀 모델 객체 (`RandomForestRegressor`, `XGBRegressor`, `GradientBoostingRegressor` 등)  
  - `X`: 전처리된 피처 데이터  
  - `y`: 타깃 변수  
- 출력:  
  - `mae`, `mse`, `rmse`, `r2` (`float`)

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import numpy as np

mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
rmse = np.sqrt(mse)
r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

return mae, mse, rmse, r2
```

## Random Forest 하이퍼파라미터 튜닝 루프

- 목적: 다양한 `RandomForest` 하이퍼파라미터 조합에 대해 평가 수행  
- 필요성: 최적의 성능을 보이는 모델 조합 탐색  
- 입력:  
  - `rf_params_list`: 사전 정의된 5가지 하이퍼파라미터 조합  
- 출력:  
  - `rf_results`: 조합별 성능 지표와 파라미터가 담긴 리스트

```python
from sklearn.ensemble import RandomForestRegressor

rf_results = []
for params in rf_params_list:
    model = RandomForestRegressor(**params, random_state=42)
    mae, mse, rmse, r2 = evaluate_model(model, X_processed, y)
    rf_results.append({'Model': 'RandomForest', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, **params})
```

## XGBoost 하이퍼파라미터 튜닝 루프

- 목적: 다양한 `XGBoost` 파라미터 조합에 대해 모델 평가  
- 필요성: 가장 낮은 오차를 주는 `XGBoost` 조합 탐색  
- 입력:  
  - `xgb_params_list`: 사전 정의된 5가지 하이퍼파라미터 조합  
- 출력:  
  - `xgb_results`: 조합별 성능 지표 리스트

```python
from xgboost import XGBRegressor

xgb_results = []
for params in xgb_params_list:
    model = XGBRegressor(**params, random_state=42)
    mae, mse, rmse, r2 = evaluate_model(model, X_processed, y)
    xgb_results.append({'Model': 'XGBoost', 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, **params})
```

## `results_df` 생성 및 최적 조합 추출

- 목적: `RandomForest`와 `XGBoost` 결과를 통합하여 비교 가능한 표 생성  
- 출력:  
  - `results_df`: 모든 모델-조합의 성능 비교표  
  - `best_result`: MAE가 가장 낮은 조합 (Series)

```python
import pandas as pd

results_df = pd.DataFrame(rf_results + xgb_results)
best_result = results_df.loc[results_df['MAE'].idxmin()]
```

## 최종 모델 학습 및 시각화

- 목적: 최적 모델로 전체 데이터를 학습하고, 피처 중요도 시각화  
- 필요성: 최적 모델로 해석 가능한 분석 결과를 제공  
- 입력:  
  - `X_processed`: 전처리된 피처 데이터  
  - `y`: 타깃 변수  
- 출력:  
  - `importance_df`: 중요도 수치를 포함한 피처 목록  
  - `barplot` 시각화 (상위 15개 피처)

```python
if best_result['Model'] == 'RandomForest':
    model = RandomForestRegressor(**{k: best_result[k] for k in rf_param_keys}, random_state=42)
else:
    model = XGBRegressor(**{k: best_result[k] for k in xgb_param_keys}, random_state=42)

model.fit(X_processed, y)
importances = model.feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Top 15 Feature Importances")
plt.show()
```

## `plot_feature_importance(model, feature_names: list, top_n=15) -> None`

- 목적: 학습된 모델의 피처 중요도를 상위 `top_n` 기준으로 시각화  
- 필요성: 모델 해석력 향상 및 비즈니스 인사이트 도출  
- 입력:  
  - `model`: 학습 완료된 모델  
  - `feature_names`: 피처 이름 리스트  
  - `top_n`: 시각화할 중요 피처 수  
- 출력:  
  - `matplotlib` barplot 출력

```python
def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f"Top {top_n} Feature Importances")
    plt.show()
```


# Salary_Modeling_Analysis

## 결측치 처리 루틴

- 목적: `avg_salary`를 포함한 수치형/범주형 변수의 결측값을 채워 학습 가능한 완전한 데이터셋을 생성  
- 필요성: 머신러닝 모델은 결측값을 허용하지 않기 때문에, 신뢰성 있는 학습을 위해 사전 결측 처리 필요  
- 입력:  
  - `df`: 전처리 전의 원본 DataFrame  
- 출력:  
  - 결측치가 모두 처리된 DataFrame (`glassdoor_cleaned_final_filled.csv`로 저장됨)  
- 처리 방식:  
  - 수치형 변수: 평균값을 반올림하여 채움  
  - 범주형 변수: 최빈값으로 채움  
  - `Position_Encoded`는 수치형이지만 범주로 처리

```python
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(round(df[col].mean()))

for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv("glassdoor_cleaned_final_filled.csv", index=False)
```

## `evaluate_model(model, X, y) -> Tuple[float, float, float, float]`

- 목적: 5-Fold Cross Validation을 통해 평균 MAE, MSE, RMSE, R² 점수를 반환  
- 필요성: 모델 간 객관적인 성능 비교 및 최적 모델 선정을 위한 핵심 평가 도구  
- 입력:  
  - `model`: 학습할 회귀 모델 (예: `RandomForestRegressor`, `XGBRegressor`)  
  - `X`: 전처리된 입력 데이터 (`np.ndarray`)  
  - `y`: 타깃 변수 (`avg_salary`)  
- 출력:  
  - `mae`: 평균 절대 오차  
  - `mse`: 평균 제곱 오차  
  - `rmse`: 평균 제곱근 오차  
  - `r2`: 결정계수 R²

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error').mean()
mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
rmse = np.sqrt(mse)
r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

return mae, mse, rmse, r2
```

## 하이퍼파라미터 튜닝 (GridSearchCV / RandomizedSearchCV)

### Random Forest (`GridSearchCV`)
- 목적: 다양한 파라미터 조합(n_estimators, max_depth, max_features)을 평가하여 최적 모델 선택  
- 평가 지표: `neg_root_mean_squared_error`  
- 출력:  
  - `best_estimator_`  
  - `best_params_`  
  - `best RMSE`

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'max_features': ['auto', 'sqrt']
}

grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid,
                    scoring='neg_root_mean_squared_error',
                    cv=5,
                    n_jobs=-1)

grid.fit(X, y)
best_model = grid.best_estimator_
```

### XGBoost (`GridSearchCV`)
- 목적: 다양한 `XGBRegressor` 조합 평가 (`n_estimators`, `learning_rate`, `max_depth`)  
- 출력:  
  - `best_estimator_`, `best_params_`  
  - `log RMSE`, `역변환된 RMSE`

```python
from xgboost import XGBRegressor

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

grid = GridSearchCV(XGBRegressor(random_state=42),
                    param_grid,
                    scoring='neg_root_mean_squared_error',
                    cv=5,
                    n_jobs=-1)

grid.fit(X, np.log1p(y))  # 로그 변환
log_rmse = -grid.best_score_
actual_rmse = np.expm1(log_rmse)
```

## 로그 변환 및 역변환 성능 평가

- 목적: `avg_salary`에 로그 변환을 적용하여 분포를 안정화하고 예측 성능을 비교  
- 처리 흐름:  
  - `y_log = np.log1p(y)`  
  - 예측 후 `np.expm1()`으로 역변환  
- 출력:  
  - 로그 스케일 RMSE / R²  
  - 역변환 후 실제 RMSE / R²

```python
y_log = np.log1p(y)
model.fit(X, y_log)
y_pred_log = model.predict(X_val)
y_pred_actual = np.expm1(y_pred_log)

rmse_log = mean_squared_error(y_log, model.predict(X_val), squared=False)
r2_log = r2_score(y_log, model.predict(X_val))

rmse_actual = mean_squared_error(y_val, y_pred_actual, squared=False)
r2_actual = r2_score(y_val, y_pred_actual)
```

## 이상치 제거 후 재학습

- 목적: `avg_salary`의 상/하위 2% 극단값을 제거하고 모델 재학습  
- 필요성: 이상치 제거를 통해 모델의 일반화 성능 향상  
- 처리 방식:  
  - `quantile([0.02, 0.98])`을 기준으로 필터링  
  - 동일한 파이프라인으로 재학습 및 평가  
- 출력:  
  - `trim_pipe`: 이상치 제거 후 모델  
  - `rmse_trim`, `r2_trim_mean`

```python
q_low, q_high = y.quantile([0.02, 0.98])
mask = (y >= q_low) & (y <= q_high)
X_trim = X[mask]
y_trim = y[mask]

trim_pipe.fit(X_trim, y_trim)
rmse_trim, r2_trim_mean = evaluate_model(trim_pipe, X_trim, y_trim)[2:]
```

## Feature Importance 시각화

- 목적: 최종 학습된 모델의 피처 중요도를 시각화하여 급여 예측에 중요한 요인을 파악  
- 입력:  
  - `final_model`: 최적 하이퍼파라미터로 학습된 모델  
  - `feature_names`: `OneHotEncoding`을 포함한 전체 피처명 리스트  
- 출력:  
  - `importance_df`: 중요도 기준 정렬된 피처 목록  
  - Top 15 `barplot` 시각화 출력

```python
importances = final_model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Top 15 Feature Importances")
plt.show()
```


# Company_Rating_Prediction

## Processing

### `extract_salary(s: str) → Tuple[float, float, float]`

- 목적: 급여 문자열에서 최소/최대/평균 급여를 수치로 추출  
- 이유: 모델 학습을 위해 문자열 형태의 급여 정보를 정량화할 필요가 있음  
- 입력:  
  - `s`: 급여 문자열 (예: `"$60K-$90K (Glassdoor est.)"`)  
- 출력:  
  - `min_salary`, `max_salary`, `avg_salary`: 각각 최소, 최대, 평균 급여 (`float`)

### `map_size(size_str: str) → str`

- 목적: 다양한 기업 규모 문자열을 `'Small'`, `'Medium'`, `'Large'`, `'Very Large'`로 단순화  
- 이유: 복잡한 범주형 데이터를 모델 학습에 적합하게 정리  
- 입력:  
  - `size_str`: 기업 규모 문자열  
- 출력:  
  - 그룹화된 범주형 크기 문자열

### `classify_domain(title: str) → str`

- 목적: 직무 제목에서 업무 도메인 분류 (예: Data Science, Marketing 등)  
- 이유: 직무별 경향 분석 및 예측 정확도 향상  
- 입력:  
  - `title`: 원시 Job Title 문자열  
- 출력:  
  - 도메인 분류 문자열

### `classify_seniority(title: str) → str`

- 목적: 직무 제목에서 직급 수준 추출 (예: Junior, Manager 등)  
- 이유: 직급에 따른 급여 및 평가 차이를 반영하기 위해  
- 입력:  
  - `title`: 원시 Job Title 문자열  
- 출력:  
  - 직급 수준 문자열

### `fill_features(row: pd.Series) → pd.Series`

- 목적: 결측된 급여 및 회사 나이를 그룹 평균 또는 회사 평균으로 보완  
- 이유: 데이터 결측치로 인한 학습 오류 방지  
- 입력:  
  - `row`: 데이터프레임의 행 (`Series`)  
- 출력:  
  - 보완된 행 (`Series`)

### `classify_ownership(val: str) → str`

- 목적: `'Type of ownership'` 컬럼을 범주형 그룹으로 분류 (`Private`, `Public`, `Public Service` 등)  
- 이유: 다양한 소유 형태를 정형화해 모델 입력으로 사용  
- 입력:  
  - `val`: 기업 소유 형태 문자열  
- 출력:  
  - 그룹화된 소유 형태 문자열

## Modeling

### `prepare_data(df: pd.DataFrame) → Tuple[X, y_reg, y_cls]`

- 목적: Glassdoor 데이터에서 회귀용(`y_reg`)과 분류용(`y_cls`) 타깃을 생성하고, 피처를 전처리하여 학습에 적합한 형태로 반환  
- 이유: 타깃 변수인 `Rating`을 기준으로 두 가지 ML task (회귀/분류)를 분리하고자 하며, 전처리를 통해 모델 학습을 안정화  
- 입력:  
  - `df`: 전처리 전의 Glassdoor `DataFrame`  
- 출력:  
  - `X`: 수치 및 범주형 전처리가 완료된 피처 행렬  
  - `y_reg`: `Rating` 연속형 타깃  
  - `y_cls`: `Rating >= 3.5`에 따른 이진 타깃 (`0/1`)

### `evaluate_linear_regression(X, y_reg, test_sizes) → List[Tuple[float, float, float]]`

- 목적: 여러 `test_size` 비율에서 선형 회귀 성능을 측정 (`R²`, `RMSE`)  
- 이유: 데이터 분할 비율이 예측력에 미치는 영향을 분석하기 위함  
- 입력:  
  - `X`: 전처리된 피처 행렬  
  - `y_reg`: 연속형 타깃 (`Rating`)  
  - `test_sizes`: 실험할 테스트셋 비율 리스트 (예: `[0.2, 0.3, 0.4, 0.5]`)  
- 출력:  
  - `[(0.2, 0.73, 0.55), ..., (0.5, 0.69, 0.60)]` 같은 (비율, R², RMSE) 튜플 리스트

### `evaluate_decision_tree(X, y_cls, max_depths) → List[Tuple[int, float]]`

- 목적: 결정 트리 모델을 다양한 깊이(`max_depth`)로 훈련 후 정확도 비교  
- 이유: 가장 일반화 성능이 좋은 트리 깊이를 탐색하여 과적합 방지  
- 입력:  
  - `X`: 전처리된 피처  
  - `y_cls`: 이진 분류 타깃  
  - `max_depths`: 실험할 트리 깊이 리스트 (예: `[3, 4, 5, 6, 7]`)  
- 출력:  
  - `[(3, 0.79), ..., (7, 0.84)]`와 같은 (깊이, 정확도) 리스트

### `final_model_evaluation_and_selection() → None`

- 목적: 최적의 `test_size` 및 `max_depth`를 기반으로 모델 선택  
  - 회귀 산점도  
  - 혼동 행렬  
  - 교차 검증 결과 등을 시각화  
- 이유: 모든 모델 결과를 종합적으로 비교·시각화하여 최종 모델을 결정  
- 입력: 없음 (이전 함수들에서 파생된 변수들 사용)  
- 출력:  
  - 그래프: `R²` 변화, `RMSE` 변화, 산점도, 혼동 행렬  
  - 텍스트 요약: 최고 `R²`, `RMSE`, 분류 정확도, 교차검증 평균

### `plot_cross_validation_results(cv_scores: np.ndarray) → None`

- 목적: 5-Fold 교차 검증 결과를 막대그래프로 시각화하고 평균선을 표시  
- 이유: 각 Fold 간의 성능 일관성을 시각적으로 검토하여 모델의 안정성 평가  
- 입력:  
  - `cv_scores`: `np.ndarray` 형태의 교차 검증 정확도 (길이 5)  
- 출력:  
  - 막대그래프 출력 (`x축`: fold 번호, `y축`: accuracy, 빨간 평균선 포함)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_cross_validation_results(cv_scores):
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(cv_scores)+1), cv_scores)
    plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--', label='Mean Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Scores')
    plt.legend()
    plt.show()
```
