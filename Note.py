'''
1. 정규화.....
    Scikit-learndml 데이터 전처리 모듈 :
        Binarizer: 임계값을 기준으로 데이터를 0과 1로 이진화합니다. 연속형 변수를 이진 변수로 변환할 때 사용합니다.
        FunctionTransformer: 임의의 함수를 적용하여 데이터를 변환합니다. 사용자 정의 변환을 적용할 때 유용합니다.
        KBinsDiscretizer: 연속형 데이터를 구간으로 나눕니다. 연속형 변수를 범주형 변수로 변환할 때 사용합니다.
        KernelCenterer: 커널 행렬을 중앙에 맞춥니다. 주로 서포트 벡터 머신에서 커널 트릭을 사용할 때 필요합니다.
        LabelBinarizer: 레이블을 원-핫 인코딩으로 변환합니다. 다중 클래스 분류 문제에서 사용합니다.
        LabelEncoder: 레이블을 0부터 n_classes-1까지의 값으로 인코딩합니다. 분류 문제의 타겟 변수를 정수로 변환할 때 사용합니다.
        MultiLabelBinarizer: 멀티 레이블 형식을 이진화합니다. 멀티 레이블 분류 문제에 사용합니다.
        MaxAbsScaler: 각 특성의 최대 절대값을 기준으로 스케일링합니다. 데이터의 스케일을 조정할 때 사용합니다.
        MinMaxScaler: 각 특성을 지정된 범위(기본적으로 0과 1)로 스케일링합니다. 데이터의 스케일을 조정할 때 널리 사용됩니다.
        Normalizer: 샘플별로 데이터를 정규화하여 단위 길이를 가지도록 합니다. 벡터의 크기만 중요할 때 사용합니다.
        OneHotEncoder: 범주형 특성을 원-핫 인코딩으로 변환합니다. 범주형 변수를 모델에 입력하기 위해 변환할 때 사용합니다.
        OrdinalEncoder: 범주형 특성을 정수 배열로 인코딩합니다. 순서가 있는 범주형 변수를 정수로 변환할 때 사용합니다.
        PolynomialFeatures: 다항식 특성과 상호작용 특성을 생성합니다. 비선형 모델을 사용할 때 유용합니다.
        PowerTransformer: 데이터를 더 정규 분포와 유사하게 만들기 위해 특성별로 거듭제곱 변환을 적용합니다. 정규 분포를 가정하는 알고리즘에 적합합니다.
        QuantileTransformer: 분위수 정보를 사용하여 특성을 변환합니다. 이상치에 덜 민감한 변환을 원할 때 사용합니다.
        RobustScaler: 이상치의 영향을 받지 않도록 통계치를 사용해 스케일링합니다. 이상치가 많은 데이터에 적합합니다.
        SplineTransformer: 특성에 대한 B-스플라인 베이스를 생성합니다. 비선형 관계를 모델링할 때 사용됩니다.
        StandardScaler: 평균을 제거하고 단위 분산으로 스케일링하여 특성을 표준화합니다. 많은 머신러닝 알고리즘이 가정하는 데이터 분포에 적합하게 만듭니다.
        TargetEncoder: 회귀와 분류 문제의 타겟 인코딩을
    
    조건
    1. 미래데이터를 포함한 정규화의 문제
    2. 

2. render 액션 마커 표시 해결하기....

3. reward 함수 정리하기

4. step 리턴 값 수정하기

5. obs_spaces 수정하기

6. 

전체 윈도우에서 obs할 윈도우를 설정하고 obs윈도우를 슬라이딩





--------------------------------------------------------------------------------------
데이터 스케일링

1. discrete한 데이터는 그대로 사용.
2. 음수값이 있는 데이터는 signed_log를 사용.
3. 나머지는 log1p를 사용.


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

obs_example_transformed = {
    "chart_data": np.log1p(np.array([[100.0, 200.0, 150.0, 120.0, 180.0],
                                     [1000.0, 1200.0, 1100.0, 1000.0, 1300.0],
                                     [0.1, 0.2, 0.15, 0.12, 0.18]])),
    "position": 0,  # Long
    "action": 3,  # Hold
    "current_price": np.log1p(np.array([1250.0])),
    "avg_price": np.log1p(np.array([1200.0])),
    "pnl": signed_log(np.array([-50.0])),  # pnl을 음수로 변경
    "total_pnl": signed_log(np.array([150.0])),
    "usdt_balance": np.log1p(np.array([1000.0])),
    "size": np.log1p(np.array([0.1])),
    "margin": np.log1p(np.array([120.0])),
    "total_balance": np.log1p(np.array([1150.0]))
}

변환 결과:
    chart_data:
        [[4.61512052 5.30390725 5.01063529 4.79579055 5.19295685]
        [6.90875478 7.09007684 7.00306546 6.90875478 7.17011843]
        [-2.30258509 -1.60943791 -1.89711998 -2.12026354 -1.7147984 ]]
    
        position: 0 (변환 없음)
        action: 3 (변환 없음)
        current_price: [7.13089883]
        avg_price: [7.09007684]
        pnl: [-3.93182563] # pnl이 음수로 변경되어 signed_log 변환됨
        total_pnl: [5.01063529]
        usdt_balance: [6.90875478]
        size: [-2.30258509]
        margin: [4.79579055]
        total_balance: [7.04752389]

        chart_data: -2.30258509 ~ 7.17011843
        current_price: 7.13089883
        avg_price: 7.09007684
        pnl: -3.93182563
        total_pnl: 5.01063529
        usdt_balance: 6.90875478
        size: -2.30258509
        margin: 4.79579055
        total_balance: 7.04752389
        
        대부분의 값들이 -2.3에서 7.2 사이의 범위로 조정
        각 스텝마다 변환하기 수월하것으로 예상됨.
        이건 논문으로 발표해야 하는것아닌가 생각이듬. ㅋㅋㅋㅋ







'''