## 회귀 및 분류 분석 코드 설명

아래는 사용자가 작성한 Python 코드를 각 섹션별로 설명한 Markdown 문서입니다.

---

### 1. 라이브러리 임포트

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_diabetes, make_classification, make_moons
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, roc_curve, auc
)
```

* **numpy, pandas**: 수치 연산 및 데이터프레임 처리
* **matplotlib**: 그래프 시각화
* **sklearn.datasets**: 당뇨병 회귀용 및 분류용 샘플 데이터 생성
* **sklearn.model\_selection**: 학습/테스트 데이터 분리
* **sklearn.linear\_model**: 선형 및 로지스틱 회귀
* **sklearn.preprocessing**: 다항 특성 생성
* **sklearn.metrics**: 평가 지표 계산 (R², MAE, RMSE, 분류 리포트, ROC·AUC)

---

### 2. `regression_analysis()` 함수

실제 당뇨병 데이터셋을 사용한 다양한 회귀 모델을 학습·평가합니다.

#### 2.1 데이터 로드 및 EDA

```python
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target
```

* `X.describe()`로 각 특성의 **기술 통계** 확인
* 히스토그램(`X.hist`) 및 박스플롯(`X.plot(kind='box')`)으로 **분포와 이상치** 시각화

#### 2.2 단순 선형 회귀 (bmi → target)

```python
X_bmi = X[['bmi']].values
X_tr, X_te, y_tr, y_te = train_test_split(...)
slr = LinearRegression().fit(X_tr, y_tr)
y_pred = slr.predict(X_te)
```

* **훈련/테스트 분리(7:3)** 후 BMI 하나만 독립 변수로 사용
* **R², MAE, RMSE** 계산 및 출력
* 산점도 위에 **회귀선** 겹쳐 그리기

#### 2.3 다중 선형 회귀 (all features)

```python
mlr = LinearRegression().fit(X_tr_all, y_tr_all)
y_pred_all = mlr.predict(X_te_all)
```

* **모든 10개 특성**을 이용해 회귀 모델 학습
* 동일 지표(R², MAE, RMSE)로 성능 평가

#### 2.4 다항 회귀 (bmi only, degree=2,3)

```python
for deg in (2,3):
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X_bmi)
    pr = LinearRegression().fit(...)
```

* BMI 특성을 **2차, 3차** 다항으로 확장
* 각 모델에 대해 **성능 지표 출력** 및 **곡선** 시각화
* 선형과 비교하여 비선형 관계 포착 여부 확인

#### 2.5 다변수 다항 회귀 (bmi & bp, degree=2)

```python
X_2 = X[['bmi','bp']]
poly2 = PolynomialFeatures(degree=2)
mpr = LinearRegression().fit(...)
```

* BMI 및 혈압(bp)을 2차 다항으로 확장
* **다변수 다항 회귀** 성능 평가

---

### 3. `under_overfitting_analysis()` 함수

합성 데이터를 이용해 **언더피팅 vs 오버피팅** 현상 분석

#### 3.1 데이터 생성 및 분할

```python
X_syn = np.sort(5 * np.random.rand(100,1), axis=0)
y_syn = np.sin(X_syn).ravel() + ...
```

* **사인 곡선 기반 합성 데이터** 생성 후 노이즈 추가
* **train\_test\_split**으로 70:30 분할

#### 3.2 다항 모델 학습 및 RMSE 계산

```python
for deg in range(1,16):
    pf = PolynomialFeatures(degree=deg)
    lr = LinearRegression().fit(...)
    train_err.append(...)
    test_err.append(...)
```

* 다항 차수(degree)별로 학습/테스트 RMSE 계산
* **차수 vs RMSE 그래프** 그려 언더피팅/적절/오버피팅 구간 시각화

---

### 4. `logistic_classification()` 함수

이진 분류용 **로지스틱 회귀** 실습

```python
X_bin, y_bin = make_classification(...)
log_bin = LogisticRegression().fit(Xb_tr, yb_tr)
```

* **make\_classification**으로 이진 분류 데이터 생성
* **classification\_report**로 정밀도, 재현율, F1, 정확도 확인
* **ROC 곡선** 및 **AUC** 계산·시각화

---

### 5. `logistic_vs_polynomial()` 함수

비선형 데이터(`make_moons`)에 대한 **기본 vs 다항 로지스틱 회귀** 비교

```python
# Base logistic
log_base = LogisticRegression().fit(...)
# Polynomial logistic (degree=3)
pf3 = PolynomialFeatures(degree=3)
log_poly = LogisticRegression(max_iter=10000).fit(...)
```

* **make\_moons**으로 비선형 분리 불가능 데이터 생성
* 기본 로지스틱과 3차 다항 변환 후 로지스틱의 **AUC 비교**
* **ROC 곡선** 시각화를 통해 모델 성능 차이 확인

---

> 이 Markdown 파일을 통해 코드의 전체 흐름과 각 섹션의 목적, 사용 방법, 주요 평가 지표를 명확히 이해할 수 있습니다.

---

### 6. 결과 분석 및 인사이트

#### 6.1 EDA 결과 요약

* **기술 통계**: 모든 특성이 평균 0, 표준편차 ≈0.0476으로 표준화됨.
* **히스토그램**: 대부분 연속형 특성(age, bmi, bp, s1\~s6)는 대략 정규분포를 따르나, `sex`와 `s4`는 두 개의 이산값을 가지는 범주형 분포 형태.
* **박스플롯**: `s1`, `s2`, `s3`, `s5`, `s6`에서 다수의 이상치(±0.1 이상) 확인. 변동성이 큰 특성은 추가 전처리(이상치 처리) 검토 필요.

#### 6.2 단순 선형 회귀 (bmi → target)

* **R²**: 0.2803
* **MAE**: 50.5931
* **RMSE**: 62.3293
* **해석**: BMI 한 개만 사용 시 타깃 변동성의 약 28% 설명. 예측 오차는 평균 약 62의 범위로, 단일 변수 모델의 한계를 보여줌.
* **시각화**: 회귀선이 데이터 중앙 경향을 따르나, 산점도 확산이 커 예측 신뢰도 낮음.

#### 6.3 다중 선형 회귀 (all features)

* **R²**: 0.4773
* **MAE**: 41.9194
* **RMSE**: 53.1202
* **해석**: 10개 특성 활용 시 설명력 약 48%로 크게 향상.
* **특성 기여도**: 회귀 계수 분석 시 `bmi`, `bp`, `s1` 순으로 영향력 큼.

#### 6.4 다항 회귀 (bmi only, degree=2,3)

* **2차**: R²=0.2766, MAE=50.6971, RMSE=62.4923
* **3차**: R²=0.2770, MAE=50.6756, RMSE=62.4737
* **해석**: 단순 선형과 거의 동일한 성능. BMI 단일 모델에 대한 다항 확장의 효과 미미.

#### 6.5 다변수 다항 회귀 (bmi & bp, degree=2)

* **R²**: 0.3354
* **MAE**: 48.5065
* **RMSE**: 59.8971
* **해석**: BMI와 혈압 상호작용을 반영해 선형 대비 예측 오차 소폭 감소.

#### 6.6 언더/오버피팅 분석

* **Train RMSE**: \[0.64,0.54,0.48,0.48,0.47,0.47,0.47,0.47,0.47,0.46,0.46,0.46,0.45,0.46,0.46]
* **Test RMSE**:  \[0.59,0.47,0.40,0.41,0.40,0.40,0.41,0.41,0.43,0.39,0.40,0.39,0.47,0.39,0.39]
* **해석**: 2\~3차 모델에서 과소적합 해소, 3차에서 Test RMSE 최저(0.40). 차수 ≥12에서 Test RMSE 급등→과적합.

#### 6.7 로지스틱 분류

* **Binary (make\_classification)**

  * Precision: 0.96 / 0.95 (클래스 0/1)
  * Recall:    0.96 / 0.95
  * F1-score: 0.96 / 0.95
  * Accuracy: 0.96
  * AUC: 0.9842
* **Base vs Polynomial (make\_moons)**

  * Base Logistic AUC: 0.9646
  * Polynomial Logistic (deg=3) AUC: 0.9884
* **해석**: 기본 모델도 양호하나, 다항 특성 추가 시 AUC 1.3%p 상승.

> **종합 인사이트**:
>
> * 다중 변수 활용 시 회귀 모델 설명력 크게 증가.
> * 비선형 항 확장은 단일 특성보다 다중 특성 조합에서 유의미.
> * 최적 차수 및 정규화 기법으로 언더/오버피팅 균형 필요.
> * 분류 모델은 다항 특성으로 ROC 성능 추가 개선 가능.
