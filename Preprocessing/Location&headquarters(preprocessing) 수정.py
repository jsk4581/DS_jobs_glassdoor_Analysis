#1. 라이브러리 불러오기
import numpy as np
import pandas as pd

#2. 데이터 불러오기
df = pd.read_csv("glassdoor_jobs.csv")

#3. 결측치 확인 후 임의로 drop하여 결측치 생성
# missing data 몇개인지 확인
missing_location = df['Location'].isnull().sum()
missing_headquarters = df['Headquarters'].isnull().sum()
print(f"Missing in 'Location': {missing_location}")
print(f"Missing in 'Headquarters': {missing_headquarters}")

# 중복 row 확인
duplicate_rows = df.duplicated().sum()
print("Duplicate Rows:", duplicate_rows)

# value 값 확인
print(df['Location'].value_counts())
print(df['Headquarters'].value_counts())

# 임의로 drop해서 missing data 만들기
np.random.seed(42)
# location 컬럼에서 임의의 15개 인덱스를 골라 nan으로 설정
location_missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[location_missing_idx, 'Location'] = np.nan

# headquarters 컬럼에서도 임의의 15개 인덱스를 골라 nan으로 설정
headquarters_missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[headquarters_missing_idx, 'Headquarters'] = np.nan

# null 개수 확인
print("After artificial NaNs:")
print(df[['Location', 'Headquarters']].isnull().sum())


#4. 데이터 정제
#dirty data 정제
def clean_location_col(col):
    return (col
            .str.strip()              # 공백 제거
            .str.lower()              # 소문자 통일
            .str.replace(', ', ',', regex=False)  # 쉼표 양쪽 공백 제거
            .str.replace(',', ', ', regex=False)) # 쉼표 뒤에만 공백 붙이기

df['Location'] = clean_location_col(df['Location'])
df['Headquarters'] = clean_location_col(df['Headquarters'])

# missing data 채우기 (최빈값 사용)
# 각 컬럼의 최빈값 계산
mode_location = df['Location'].mode()[0]
mode_headquarters = df['Headquarters'].mode()[0]

# 결측치 채우기
df['Location'] = df['Location'].fillna(mode_location)
df['Headquarters'] = df['Headquarters'].fillna(mode_headquarters)

# 확인
print(df[['Location', 'Headquarters']].isnull().sum())


# 5. 본사근무 여부 피처 생성 (state 기준)
# location에서 state 추출
df['Location_state'] = df['Location'].apply(lambda x: x.split(',')[1].strip() if ',' in x else 'Unknown')
# headquarters에서 state 추출
df['Headquarters_state'] = df['Headquarters'].apply(lambda x: x.split(',')[1].strip() if ',' in x else 'Unknown')

# state 기준으로 binning (상위 6개 + 나머지는 'Other'로)
top5_location_state = df['Location_state'].value_counts().nlargest(6).index.tolist()
top5_headquarters_state = df['Headquarters_state'].value_counts().nlargest(6).index.tolist()

df['Location_state_binned'] = df['Location_state'].apply(lambda x: x if x in top5_location_state else 'Other')
df['Headquarters_state_binned'] = df['Headquarters_state'].apply(lambda x: x if x in top5_headquarters_state else 'Other')

# location과 headquarters가 같으면 본사 근무 (Yes), 아니면 No
df['works_at_headquarters'] = df.apply(
    lambda row: 'Yes' if row['Location'] == row['Headquarters'] else 'No', axis=1
)

# 결과 확인
print(df[['Location', 'Headquarters', 'works_at_headquarters']].head(10))

# 7. 전처리 (Binning): region 기준으로 전처리
region_map = {
    'Northeast': ['ny', 'nj', 'ma', 'pa', 'ct', 'ri', 'vt', 'nh', 'me'],
    'Midwest': ['il', 'oh', 'mi', 'wi', 'in', 'mo', 'mn', 'ia', 'ks', 'ne'],
    'South': ['tx', 'fl', 'ga', 'nc', 'va', 'tn', 'al', 'sc', 'la', 'ky', 'ok', 'ms', 'ar'],
    'West': ['ca', 'wa', 'or', 'co', 'az', 'nv', 'ut', 'nm', 'id', 'mt', 'wy', 'ak', 'hi']
}

def map_state_to_region(state):
    for region, states in region_map.items():
        if state in states:
            return region
    return 'Other'  # 알 수 없거나 희귀한 state

# location_state, headquarters_state → region
df['Location_region'] = df['Location_state'].apply(map_state_to_region)
df['Headquarters_region'] = df['Headquarters_state'].apply(map_state_to_region)

print(df['Location_region'].value_counts())
print(df['Headquarters_region'].value_counts())


