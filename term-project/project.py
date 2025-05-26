import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 

df = pd.read_csv("/Users/q/IdeaProjects/data-science/term-project/survey.csv")
df


def handling_missing_value(df):
    """
    결측치를 시각화하고, 결측치를 처리한 후 결과를 반환.

    1. 결측치가 있는 컬럼을 바 차트로 시각화 (결측치 수 기준)
    2. 특정 컬럼에 대해 규칙 기반 결측치 대치 수행:
      - state: 미국이면 최빈값으로, 미국이 아니면 NaN으로
      - work_interfere: 결측치는 "Don't know"로 대체
      - self_employed: 최빈값으로 대체

    Args: df(pd.DataFrame): 원본 데이터프레임
    Returns: pd.DataFrame(결측치 처리된 데이터프레임)
    """

    # 결측치 수 계산
    missing = df.isnull().sum()
    # 결측치가 0보다 큰 컬럼만 추출
    missing = missing[missing > 0]
    # 결측치 수를 기준으로 내림차순 정렬
    missing = missing.sort_values(ascending=False)

    # 결과 출력
    print("Before:")
    print(missing)
    # 시각화
    plt.figure(figsize=(10, 6))
    missing.plot(kind="bar")
    plt.xlabel("Column")
    plt.xticks(rotation=45)
    plt.ylabel("Missing Values")
    plt.title("Missing Values in Each columns")
    plt.show()

    """결측값 대치"""

    # 안전성을 위해 df 복사본으로 전처리 진행
    df_prep = df.copy()

    # 미국인데, state값이 없는 행 최빈값으로 대치
    us_mode = df_prep[df_prep["Country"] == "United States"]["state"].mode()[0]
    df_prep.loc[
        (df_prep["Country"] == "United States") & (df_prep["state"].isnull()), "state"
    ] = us_mode
    # 미국이 아닌데도 state 값이 있는 행 결측치 처리
    df_prep.loc[
        (df_prep["Country"] != "United States") & (~df_prep["state"].isnull()), "state"
    ] = np.nan

    # work_interfere 결측값 처리: null 값을 don't know로 변환
    work_interfere_null_idx = df_prep[df_prep["work_interfere"].isnull()].index
    df_prep.loc[work_interfere_null_idx, "work_interfere"] = "Don't know"

    # self_employed 최빈값으로 대치
    self_employed_null_index = df_prep[df_prep["self_employed"].isnull()].index
    self_employed_mode = df_prep["self_employed"].mode()[0]
    df_prep.loc[self_employed_null_index, "self_employed"] = self_employed_mode

    # 결측치 수 계산
    missing = df_prep.isnull().sum()
    # 결측치가 0보다 큰 컬럼만 추출
    missing = missing[missing > 0]
    # 결측치 수를 기준으로 내림차순 정렬
    missing = missing.sort_values(ascending=False)

    # 결과 출력
    print("After:")
    print(missing)
    # 시각화
    plt.figure(figsize=(10, 6))
    missing.plot(kind="bar")
    plt.xlabel("Column")
    plt.xticks(rotation=45)
    plt.ylabel("Missing Values")
    plt.title("Missing Values after handling")
    plt.show()

    return df_prep


def handling_numeric_outlier(df):
    """
    이상치를 시각화하고, 이상치를 처리한 후 결과를 반환.

    1. Age 컬럼의 이상치를 시각화 (Box plot, Histogram)
    2. Age 컬럼의 이상치를 처리

    Args: df(pd.DataFrame): 원본 데이터프레임
    Returns: pd.DataFrame(이상치 처리된 데이터프레임)
    """

    # Age 이상치 확인
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].boxplot(df["Age"])
    axes[0].set_title("Boxplot of Age (Original)")
    axes[0].set_ylabel("Age")

    axes[1].hist(df["Age"], bins=20)
    axes[1].set_title("Histogram of Age (Original)")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Frequency")

    plt.show()

    """이상치 처리"""
    df_prep = df.copy()

    Q3 = df_prep["Age"].quantile(0.75)
    Q1 = df_prep["Age"].quantile(0.25)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_prep = df_prep[(df_prep["Age"] >= lower_bound) & (df_prep["Age"] <= upper_bound)]

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    axes[0].boxplot(df_prep["Age"])
    axes[0].set_title("Boxplot of Age (After handling)")
    axes[0].set_ylabel("Age")

    axes[1].hist(df_prep["Age"], bins=20)
    axes[1].set_title("Histogram of Age (After handling)")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Frequency")

    plt.show()

    return df_prep


def clean_gender_column(df):
    """
    Gender 컬럼을 정리하여 'Male', 'Female', 'Other'로 분류.

    Args: df(pd.DataFrame): 원본 데이터프레임
    Returns: pd.DataFrame(Gender 정리된 데이터프레임)
    """

    df_prep = df.copy()

    def classify_gender(gender):
        gender = str(gender).strip().lower()

        male_keywords = [
            "male",
            "m",
            "man",
            "cis male",
            "malr",
            "msle",
            "maile",
            "cis man",
            "make",
        ]
        female_keywords = [
            "female",
            "f",
            "woman",
            "cis female",
            "femake",
            "femail",
            "cis-female",
        ]

        if any(keyword in gender for keyword in male_keywords):
            return "Male"
        elif any(keyword in gender for keyword in female_keywords):
            return "Female"
        else:
            return "Other"

    df_prep["Gender_cleaned"] = df_prep["Gender"].apply(classify_gender)

    # 시각화
    print(df_prep["Gender_cleaned"].value_counts())

    plt.figure(figsize=(10, 6))
    df_prep["Gender_cleaned"].value_counts().plot(kind="bar")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.title("Cleaned Gender Distribution")
    plt.show()

    return df_prep


def EDA(df):
    """
    각 피쳐들의 분포 및 비율을 확인하는 EDA 함수.

    1. 각 컬럼을 bar, pie로 시각화.

    Args: df(pd.DataFrame): 원본 데이터프레임
    """

    region_cols = ["Country", "state"]
    employment_cols = ["self_employed", "no_employees", "remote_work", "tech_company"]
    mental_health_cols = ["family_history", "treatment", "mental_vs_physical"]
    workplace_support_cols_1 = [
        "work_interfere",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "leave",
    ]
    workplace_support_cols_2 = [
        "mental_health_consequence",
        "phys_health_consequence",
        "coworkers",
        "supervisor",
        "mental_health_interview",
        "phys_health_interview",
        "obs_consequence",
    ]

    def EDA_visualization(df, cols):

        fig, axes = plt.subplots(len(cols), 2, figsize=(19, 4 * len(cols)))

        for i, col in enumerate(cols):
            axes[i, 0].bar(df[col].value_counts().index, df[col].value_counts().values)
            axes[i, 0].set_title(col)
            axes[i, 0].set_ylabel("Count")
            axes[i, 0].tick_params(axis="x", rotation=45)

            axes[i, 1].pie(
                df[col].value_counts(),
                labels=df[col].value_counts().index,
                autopct="%1.1f%%",
            )
            axes[i, 1].set_title(col)

        plt.tight_layout()
        plt.show()

    EDA_visualization(df, region_cols)
    EDA_visualization(df, employment_cols)
    EDA_visualization(df, mental_health_cols)
    EDA_visualization(df, workplace_support_cols_1)
    EDA_visualization(df, workplace_support_cols_2)

def preprocess_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 전처리 및 피처 선택을 수행합니다.

    1. 결측치, 이상치, 성별 정리
    2. 불필요 컬럼 제거, Age 표준화
    3. Country, state, Gender 재정의
    4. 이진 Yes/No 컬럼 → 0/1 매핑
    5. 다중 범주형 One-Hot 인코딩
    6. VarianceThreshold로 분산 < 0.01인 피처 제거
    7. 상위 10개 피처 분산 바 차트 시각화

    Args:
        df (pd.DataFrame): 전처리 전 데이터프레임
    Returns:
        pd.DataFrame: 선택된 피처만 남은 데이터프레임
    """

    # 불필요한 컬럼 제거
    columns_to_drop = ["Timestamp", "comments"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Age 정규화
    scaler = StandardScaler()
    df["Age"] = scaler.fit_transform(df[["Age"]])

    print("Age 정규화")
    print(df["Age"].head(5))

    # Country가 United States가 아니면 Other로 대체
    df["Country"] = df["Country"].apply(
        lambda x: "United States" if x == "United States" else "Other"
    )

    print("Country 분포:")
    print(df["Country"].value_counts())

    # 미국외 나라의 주는 N/A로 표기
    df["state"] = df.apply(
        lambda row: row["state"] if row["Country"] == "United States" else "N/A",
        axis=1
    )

    df["Gender"] = df["Gender_cleaned"]

    # 1. 이진형 Yes/No 컬럼만 추출
    binary_cols = [
        col for col in df.columns
        if df[col].dropna().nunique() == 2 and set(df[col].dropna().unique()) <= {"Yes", "No"}
    ]

    # 2. 이진형 컬럼을 숫자 0/1로 변환
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

    # 3. 나머지 범주형 컬럼들 중에서 3개 이상 클래스가 있는 컬럼 추출
    # 대상은 범주형 타입 컬럼 중 이진형 컬럼에 없는 것
    multi_cat_cols = [
        col for col in df.select_dtypes(include="object").columns
        if col not in binary_cols
    ]

    # 4. 다중 범주형 컬럼 One-Hot Encoding (drop_first로 차원 감소)
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # 확인
    print("이진형 컬럼:", binary_cols)
    print("다중 범주형 인코딩된 컬럼:", multi_cat_cols)

    # Variance Thresholding으로 피쳐 셀렉션
    selector = VarianceThreshold(threshold=0.01)
    selected_data = selector.fit_transform(df)
    selected_features = df.columns[selector.get_support(indices=True)]
    df_selected = pd.DataFrame(selected_data, columns=selected_features)
    
    print("Variance Thresholding")
    print(f"원래 컬럼 수: {df.shape[1]}")
    print(f"선택된 컬럼 수: {df_selected.shape[1]}")
    print(f"제거된 컬럼 수: {df.shape[1] - df_selected.shape[1]}")
    print("\n선택된 피처 예시 (상위 10개):")
    print(selected_features[:10].tolist())

    # 4. 상위 10개 분산 시각화
    variances = selector.variances_
    selected_mask = selector.get_support()
    selected_variances = variances[selected_mask]
    top_indices = selected_variances.argsort()[::-1][:10]
    top_features = selected_features[top_indices]
    top_variances = selected_variances[top_indices]

    plt.figure(figsize=(12, 6))
    plt.barh(top_features[::-1], top_variances[::-1])
    plt.xlabel("Variance")
    plt.title("Top 10 Features by Variance (after threshold > 0.01)")
    plt.tight_layout()
    plt.show()

    return df_selected



def clustering_and_profiling(df: pd.DataFrame) -> pd.DataFrame:
    """
    KMeans 군집화를 수행하고, 군집 수 선정, 프로파일링 정보를 출력합니다.

    1. Elbow Method와 Silhouette Score로 최적 k 탐색 (2~10)
    2. k=3으로 모델 학습, 라벨링
    3. 군집별 샘플 수 출력
    4. 각 군집 센트로이드(특성 평균) 출력
    5. 주요 피처(treatment 등) 군집별 평균값 출력
    6. Silhouette sample 분석 및 군집별 평균 Silhouette Score
    7. treatment 교차검증 결과 출력

    Args:
        df (pd.DataFrame): 피처 선택 완료 데이터프레임
    Returns:
        pd.DataFrame: cluster, silhouette 컬럼 추가된 데이터
    """

    # 5-1) 최적의 k 탐색 (Elbow Method + Silhouette Score)
    inertia, sil_scores = [], []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df_selected)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(df_selected, labels))

    # 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(2,11), inertia, 'o-')
    axes[0].set(title='Elbow Method', xlabel='k', ylabel='Inertia')
    axes[1].plot(range(2,11), sil_scores, 'o-')
    axes[1].set(title='Silhouette Scores', xlabel='k', ylabel='Silhouette Score')
    plt.tight_layout()
    plt.show()

    # 5-2) 최적의 k 값으로 학습 (k=3)
    k_opt = 3
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_selected)

    # 5-3) DataFrame에 클러스터 라벨 추가
    df_selected['cluster'] = cluster_labels

    # 5-4) 클러스터별 샘플 수 확인
    print("Cluster sizes:")
    print(df_selected['cluster'].value_counts(), "\n")

    # 5-5) 클러스터 센트로이드(특성별 평균 위치) 확인
    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=df_selected.drop('cluster', axis=1).columns
    )
    centroids.index.name = 'cluster'
    print("Centroids (cluster × feature):")
    print(centroids.head(), "\n")

    # 5-6) 주요 피처로 클러스터 프로파일링
    profile_features = ['treatment', 'family_history', 'leave_Very difficult', 'anonymity_Yes']
    print("Cluster profiling (mean values):")
    print(df_selected.groupby('cluster')[profile_features].mean(), "\n")

    # 5-7) Silhouette sample 분석 (optional)
    sil_samp = silhouette_samples(df_selected.drop('cluster', axis=1), df_selected['cluster'])
    df_selected['silhouette'] = sil_samp
    print("Average silhouette score per cluster:")
    print(df_selected.groupby('cluster')['silhouette'].mean(), "\n")

    # 5-8) treatment 레이블과의 교차검증
    if 'treatment' in df_selected.columns:
        ct = pd.crosstab(df_selected['cluster'], df_selected['treatment'])
        print("Crosstab cluster × treatment:")
        print(ct)

    return df


def classification_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """
    치료 경험(treatment) 예측을 위한 분류 모델을 5-폴드 교차검증으로 평가합니다.

    1. feature matrix X, target y(treatment) 정의
    2. StratifiedKFold(n_splits=5)로 교차검증 설정
    3. LogisticRegression, RandomForestClassifier 모델 학습 및 검증
    4. Accuracy, Precision, Recall, F1-score 계산 및 출력

    Args:
        df (pd.DataFrame): cluster, silhouette 포함된 데이터프레임
    Returns:
        pd.DataFrame: 모델별 CV 평균 성능 지표
    """
    
    X = df_selected.drop(columns=['cluster','silhouette','treatment'], errors='ignore')
    y = df_selected['treatment']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        cv_res = cross_validate(
            model, X, y, cv=skf,
            scoring=['accuracy','precision','recall','f1'],
            return_train_score=False
        )
        results[name] = {metric: cv_res[f'test_{metric}'].mean()
                        for metric in ['accuracy','precision','recall','f1']}

    cv_df = pd.DataFrame(results).T
    cv_df.columns = ['Accuracy','Precision','Recall','F1-score']
    print("\n=== Classification CV Results ===")
    print(cv_df)  

    # 5) 전체 데이터로 최종 Logistic Regression 모델 학습
    final_lr = LogisticRegression(max_iter=1000, random_state=42)
    final_lr.fit(X, y)
    coefs = pd.Series(final_lr.coef_[0], index=X.columns)
    top_positive = coefs.sort_values(ascending=False).head(5)
    top_negative = coefs.sort_values().head(5)
    print("Top positive features for predicting treatment:")
    print(top_positive)
    print("Top negative features for predicting treatment:")
    print(top_negative)

    return cv_df
    

# 8) 대시보드 시각화
# 8) Dashboard Plotting
def plot_dashboard(cluster_sizes: list,treatment_yes: np.ndarray,treatment_no: np.ndarray,cluster_profile: pd.DataFrame,cv_results: pd.DataFrame,cluster_names: list):
    """
    컴포지트 대시보드를 생성합니다:
    1. 파이 차트: 군집별 크기 분포
    2. 스택형 막대 차트: 치료 경험 비율
    3. 레이더 차트: 군집별 주요 피처 프로파일
    4. 히트맵: 분류 모델 교차검증 지표

Args:
    cluster_sizes (list): 군집별 샘플 수
    treatment_yes (np.ndarray): 각 군집의 치료 경험(Yes) 수
    treatment_no (np.ndarray): 각 군집의 비경험(No) 수
    cluster_profile (pd.DataFrame): 군집별 평균 피처 값
    cv_results (pd.DataFrame): 분류 모델 교차검증 지표
    cluster_names (list): 군집 라벨 리스트 (예: ['High Risk','Low Risk','Mid Risk'])
"""
    pct_yes=treatment_yes/(treatment_yes+treatment_no)
    pct_no=1-pct_yes
    fig=plt.figure(figsize=(14,10))
    gs=fig.add_gridspec(2,2,wspace=0.3,hspace=0.3)
    ax0=fig.add_subplot(gs[0,0])
    ax0.pie(cluster_sizes,labels=cluster_names,autopct='%1.1f%%',startangle=140)
    ax0.set_title('Cluster Size Distribution')
    ax1=fig.add_subplot(gs[0,1])
    x=np.arange(len(cluster_names))
    ax1.bar(x,pct_no,0.5,label='No',bottom=0)
    ax1.bar(x,pct_yes,0.5,label='Yes',bottom=pct_no)
    ax1.set_xticks(x); ax1.set_xticklabels(cluster_names)
    ax1.set_ylabel('Proportion'); ax1.set_title('Treatment Experience by Cluster')
    ax1.legend()
    ax2=fig.add_subplot(gs[1,0],polar=True)
    cats=list(cluster_profile.columns)
    N=len(cats)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    for idx,name in enumerate(cluster_names):
        vals=cluster_profile.iloc[idx].tolist()+[cluster_profile.iloc[idx].tolist()[0]]
        ax2.plot(angles,vals,label=name)
        ax2.fill(angles,vals,alpha=0.25)
    ax2.set_xticks(angles[:-1]); ax2.set_xticklabels(cats)
    ax2.set_title('Cluster Profile Radar Chart')
    ax2.legend(loc='upper right',bbox_to_anchor=(1.3,1.1))
    ax3=fig.add_subplot(gs[1,1])
    im=ax3.imshow(cv_results.values,aspect='auto',cmap='viridis')
    ax3.set_xticks(np.arange(len(cv_results.columns))); ax3.set_xticklabels(cv_results.columns)
    ax3.set_yticks(np.arange(len(cv_results.index))); ax3.set_yticklabels(cv_results.index)
    for i in range(len(cv_results.index)):
        for j in range(len(cv_results.columns)):
            ax3.text(j,i,f"{cv_results.values[i,j]:.2f}",ha='center',va='center',color='white')
    ax3.set_title('Classification CV Metrics (Heatmap)')
    plt.colorbar(im,ax=ax3,fraction=0.046,pad=0.04)
    plt.show()

df = handling_missing_value(df)
df = handling_numeric_outlier(df)
df = clean_gender_column(df)

EDA(df)

df_selected = preprocess_and_select(df)
df_clustered = clustering_and_profiling(df_selected)
cv_results = classification_evaluation(df_clustered)

cluster_sizes = df_clustered['cluster'].value_counts().sort_index().tolist()
treatment_counts = df_clustered.groupby('cluster')['treatment'].value_counts().unstack(fill_value=0)
treatment_yes = treatment_counts[1].values
treatment_no = treatment_counts[0].values
cluster_profile = df_clustered.groupby('cluster')[['treatment', 'family_history', 'leave_Very difficult', 'anonymity_Yes']].mean()
cluster_names = ['High Risk', 'Low Risk', 'Mid Risk']

plot_dashboard(
    cluster_sizes,
    treatment_yes,
    treatment_no,
    cluster_profile,
    cv_results,
    cluster_names
)