import pandas as pd
import numpy as np
import re

# 데이터 로드
df = pd.read_csv('glassdoor_jobs.csv')



# ─────────────────────────────
# Job Title 전처리
# ─────────────────────────────

# 직무 도메인 분류 및 직급 분류
domain_keywords = {
    'Data Science': ['data scientist', 'data science'],
    'Data Engineering': ['data engineer', 'big data engineer', 'etl', 'data platform', 'data systems'],
    'Data Analysis': ['data analyst', 'business intelligence', 'analytics'],
    'Machine Learning': ['machine learning', 'ml engineer', 'deep learning', 'ai', 'artificial intelligence', 'nlp'],
    'Research': ['research scientist', 'research', 'r&d'],
    'Medical/Healthcare': ['medical', 'clinical', 'healthcare', 'pharmacovigilance', 'lab scientist', 'laboratory'],
    'Marketing': ['marketing', 'digital marketing', 'ecommerce'],
    'Software Engineering': ['software engineer', 'developer', 'devops', 'programmer'],
    'Business': ['business', 'consultant', 'account', 'revenue', 'risk', 'insurance', 'manager', 'director', 'executive', 'officer']
}

seniority_keywords = {
    'Chief/Head': ['chief', 'head', 'director', 'officer', 'vp', 'vice president', 'president', 'cxo'],
    'Principal': ['principal', 'distinguished'],
    'Leader': ['leader', 'lead'],  # lead 수정 사항 반영
    'Manager': ['manager'],
    'Senior': ['senior', 'sr.', 'sr ', 'sr'],
    'Junior': ['junior', 'jr.', 'jr ', 'associate', 'assistant', 'trainee', 'intern', 'entry'],
    'Staff': ['staff']
}

def classify_domain(title):
    title_lower = str(title).lower()
    for domain, keywords in domain_keywords.items():
        if any(k in title_lower for k in keywords):
            return domain
    return 'Others'

def classify_seniority(title):
    title_lower = str(title).lower()
    for level, keywords in seniority_keywords.items():
        if any(k in title_lower for k in keywords):
            return level
    return 'MidLevel'

df['Job Title'] = df['Job Title'].apply(lambda x: f"{classify_domain(x)} - {classify_seniority(x)}")


# Job Title 열 분리
split_cols = df['Job Title'].str.split(' - ', expand=True)
df['Job Title'] = split_cols[0]
df['Position'] = split_cols[1]

# 오디널 인코딩 (직급)
ordinal_map = {
    'Junior': 1,
    'MidLevel': 2,
    'Staff': 3,
    'Senior': 4,
    'Manager': 5,
    'Leader': 6,
    'Principal': 7
}

df['Position_Encoded'] = df['Position'].map(ordinal_map)



# ─────────────────────────────────────────────
# Revenue 전처리
# ─────────────────────────────────────────────
def revenue_to_numeric(revenue):
    if isinstance(revenue, str):
        revenue = revenue.lower()
        match = re.match(r'\$(\d+\.?\d*)\s*to\s*\$(\d+\.?\d*)\s*(million|billion)', revenue)
        if match:
            start, end, unit = match.groups()
            start = float(start)
            end = float(end)
            multiplier = 1_000_000 if unit == 'million' else 1_000_000_000
            return int(((start + end) / 2) * multiplier)
        match = re.match(r'\$(\d+\.?\d*)\+\s*billion', revenue)
        if match:
            return int(float(match.group(1)) * 1_000_000_000)
        if 'less than $1 million' in revenue:
            return 500_000
    return np.nan

df['Revenue'] = df['Revenue'].apply(revenue_to_numeric)



# ─────────────────────────────────────────────
# Salary Estimate 전처리
# ─────────────────────────────────────────────
def extract_salary(s):
    if pd.isnull(s) or '-1' in s:
        return np.nan, np.nan, np.nan

    s = s.lower()
    s = s.replace('employer provided salary:', '')
    s = s.replace('per hour', '')
    s = re.sub(r'\(.*?\)', '', s)  # 괄호 안 제거
    s = s.replace('$', '').replace('k', '').strip()

    try:
        min_sal, max_sal = map(int, s.split('-'))
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    except:
        return np.nan, np.nan, np.nan

# Apply to Salary Estimate
df[["min_salary", "max_salary", "avg_salary"]] = df["Salary Estimate"].apply(
    lambda x: pd.Series(extract_salary(x))
)



# ─────────────────────────────────────────────
# Size 전처리 (원하는 그룹 기준 적용)
# ─────────────────────────────────────────────
def map_size(size_str):
    if pd.isnull(size_str) or size_str in ['-1', 'Unknown']:
        return np.nan
    size_str = size_str.strip()
    if size_str in ['0 to 50 employees', '51 to 200 employees']:
        return 'Small'
    elif size_str in ['201 to 500 employees', '501 to 1000 employees']:
        return 'Medium'
    elif size_str in ['1001 to 5000 employees', '5001 to 10000 employees']:
        return 'Large'
    elif size_str == '10000+ employees':
        return 'Very Large'
    else:
        return np.nan

df["Size_cleaned"] = df["Size"].apply(map_size)



# ─────────────────────────────────────────────
# Founded 전처리 (회사 나이 계산)
# ─────────────────────────────────────────────
df["Founded_cleaned"] = df["Founded"].apply(lambda x: x if x > 1800 else np.nan)
df["Company_age"] = df["Founded_cleaned"].apply(lambda x: 2025 - x if pd.notnull(x) else np.nan)



# ─────────────────────────────────────────────
# Location & Headquarters 전처리
# ─────────────────────────────────────────────

# 임의로 drop해서 missing data 만들기
np.random.seed(42)

# location 컬럼에서 임의의 15개 인덱스를 골라 nan으로 설정
location_missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[location_missing_idx, 'Location'] = np.nan

# headquarters 컬럼에서도 임의의 15개 인덱스를 골라 nan으로 설정
headquarters_missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[headquarters_missing_idx, 'Headquarters'] = np.nan

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


# 본사근무 여부 피처 생성 (state 기준)

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

# Binning: region 기준으로 전처리
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



# ─────────────────────────────────────────────
# Inderustry & Sector 전처리
# ─────────────────────────────────────────────

# "-1" 표기를 NaN으로 변환
df.replace({'Industry': {'-1': np.nan},
            'Sector'  : {'-1': np.nan}}, 
           inplace=True)

# 회사별 Mode 계산, 회사별로 존재하는 공고의 최빈값.
ind_mode_by_co = (
    df.groupby('Company Name')['Industry']
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)
sec_mode_by_co = (
    df.groupby('Company Name')['Sector']
      .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

# 회사별 Mode로 결측치 채우기 (기존 컬럼 덮어쓰기)
df['Industry'] = df.apply(
    lambda r: ind_mode_by_co[r['Company Name']] 
              if pd.isna(r['Industry']) else r['Industry'],
    axis=1
)
df['Sector'] = df.apply(
    lambda r: sec_mode_by_co[r['Company Name']] 
              if pd.isna(r['Sector']) else r['Sector'],
    axis=1
)

# 남은 NaN → 전역 Mode(최빈값)로 채우기, 만약 회사별 공고가 없을 경우 모든 공고의 최빈값으로 채움
global_ind_mode = df['Industry'].mode()[0]
global_sec_mode = df['Sector'].mode()[0]
df.fillna({'Industry': global_ind_mode,
           'Sector'  : global_sec_mode},
          inplace=True)

# 6) Sector를 6개 카테고리로 매핑 (기존 컬럼 덮어쓰기)
sector_mapping = {
    # 1. 금융·컨설팅
    'Finance': 'Finance & Consulting',
    'Insurance': 'Finance & Consulting',
    'Accounting & Legal': 'Finance & Consulting',
    'Business Services': 'Finance & Consulting',
    # 2. IT·테크
    'Information Technology': 'Technology',
    'Telecommunications': 'Technology',
    # 3. 헬스케어·교육
    'Health Care': 'Healthcare & Education',
    'Biotech & Pharmaceuticals': 'Healthcare & Education',
    'Education': 'Healthcare & Education',
    # 4. 소비재·리테일
    'Retail': 'Consumer & Retail',
    'Consumer Services': 'Consumer & Retail',
    # 5. 산업·에너지
    'Manufacturing': 'Industrial & Energy',
    'Oil, Gas, Energy & Utilities': 'Industrial & Energy',
    'Mining & Metals': 'Industrial & Energy',
    'Construction, Repair & Maintenance': 'Industrial & Energy',
    'Aerospace & Defense': 'Industrial & Energy',
    'Transportation & Logistics': 'Industrial & Energy',
    # 6. 기타 서비스
    'Media': 'Other Services',
    'Real Estate': 'Other Services',
    'Government': 'Other Services',
    'Non-Profit': 'Other Services',
    'Arts, Entertainment & Recreation': 'Other Services',
    'Agriculture & Forestry': 'Other Services',
    'Travel & Tourism': 'Other Services'
}

df['Sector'] = df['Sector'].map(sector_mapping).fillna('Other Services')



# ─────────────────────────────────────────────
# Type of Ownership 전처리
# ─────────────────────────────────────────────

# 최빈값 계산 (결측 대체용)
most_common_ownership = df['Type of ownership'].mode()[0]

# 전처리 함수 정의
def classify_ownership(value):
    # - "-1", "Unknown", "Other Organization"은 실제 기업 구조 파악 불가
    # - 단순히 제거하거나 Others로 둘 경우 분석/모델링에서 해석성 및 예측 안정성 저하
    # - 최빈값으로 대체하여 데이터 누락 없이 편입
    if value in ["-1", "Unknown", "Other Organization"]:
        value = most_common_ownership

    # Private 그룹
    # - "Company - Private": 사기업
    # - "Subsidiary or Business Segment": 대부분 상위 사기업 소속
    # - "Contract": 사기업 중심의 고용 형태
    # - "Private Practice / Firm": 소규모 독립 사기업으로 판단
    # 따라서 이들은 "Private"으로 통합
    if value in [
        "Company - Private", "Subsidiary or Business Segment",
        "Contract", "Private Practice / Firm"
    ]:
        return "Private"

    # Public 그룹
    # - "Company - Public": 상장 대기업 또는 주식 공개 기업
    # 독립적인 경제적 규모와 해석을 위해 그룹 유지
    elif value == "Company - Public":
        return "Public"

    # Public Service 그룹
    # - "Nonprofit Organization": 공익 목적
    # - "Government": 공공 부문
    # - "Hospital": 보건 서비스
    # - "College / University", "School / School District": 교육 기관
    # 모두 공공 목적 중심으로, 유사한 고용 특성 고려
    elif value in [
        "Nonprofit Organization", "Government",
        "Hospital", "College / University", "School / School District"
    ]:
        return "Public Service"

# 새 피쳐 생성
df['Ownership_Grouped'] = df['Type of ownership'].apply(classify_ownership)



# ─────────────────────────────────────────────
# Rating 전처리
# ─────────────────────────────────────────────

# -1.0을 결측치로 간주하고 평균값으로 대체, 분포가 왜곡되지 않았으므로 안전
df['Rating'] = df['Rating'].replace(-1.0, np.nan)
rating_mean = df['Rating'].mean(skipna=True)
df['Rating'] = df['Rating'].fillna(rating_mean)



# ─────────────────────────────────────────────
# 사용할 피쳐 리스트 정의 (미사용 피쳐는 주석 처리)
# ─────────────────────────────────────────────

final_features = [
    'Job Title',                # ✔ Position 분리 후 직무명
    'Position_Encoded',         # ✔ 오디널 인코딩된 직급
    # 'Position',               # → Position_Encoded로 대체됨

    'min_salary',               # ✔ 수치형 최저 연봉
    'max_salary',               # ✔ 수치형 최고 연봉
    'avg_salary',               # ✔ 수치형 평균 연봉
     # 'Salary Estimate',       # → 위 세 변수로 분해됨
    
    'Rating',                   # ✔ 결측 보완된 수치형 평점

    'Location_state_binned',    # ✔ 정제된 지역 (상위 6개 + 기타)
    'Headquarters_state_binned',# ✔ 정제된 본사 지역 (상위 6개 + 기타)
    'works_at_headquarters',    # ✔ 본사 근무 여부 (Yes/No)
    #'Location_region',         # → 정제된 지역 (상위 6개 + 기타)
    #'Headquarters_region',     # → 정제된 본사 지역 (상위 6개 + 기타)
    # 'Location',               # → 정제 후 위 컬럼들로 분해됨
    # 'Headquarters',           # → 동일
    # 'Location_state',         # → 동일
    # 'Headquarters_state',     # → 동일

    'Size_cleaned',             # ✔ 정제된 회사 규모
    # 'Size',                   # → Size_cleaned로 대체됨

    'Company_age',              # ✔ 회사 나이 (2025 - Founded)
    # 'Founded_cleaned',        # → 위로 대체됨
    # 'Founded',                # → 동일

    'Ownership_Grouped',        # ✔ 정제된 소속 그룹
    # 'Type of ownership',      # → Ownership_Grouped로 대체됨

    'Industry',                  # ✔ 산업군 (세분화된 업종)

    'Sector',                    # ✔ 6개 대분류 부문 그룹


    #--!결측 처리 모델링 필요!--
    'Revenue',                   # ✔ 수치형 매출 규모  
]

df[final_features].to_csv("glassdoor_cleaned.csv", index=False)
print("전처리된 주요 피쳐만 포함하여 저장 완료.")
