import pandas as pd
import numpy as np
import re

# 1. 데이터 로드
df = pd.read_csv("glassdoor_jobs.csv")  # Glassdoor 데이터 CSV 파일 불러오기

# 2. Salary Estimate 파싱 함수 정의
def extract_salary(s):
    # 결측값 또는 '-1' 처리
    if pd.isnull(s) or s == '-1' or '-1' in str(s):
        return np.nan, np.nan, np.nan
    # 불필요한 텍스트 제거
    s = str(s).lower().replace('employer provided salary:', '').replace('per hour', '')
    s = re.sub(r'\(.*?\)', '', s).replace('$', '').replace('k', '').strip()
    # 최소, 최대, 평균 연봉 계산
    try:
        min_sal, max_sal = map(int, s.split('-'))
        avg_sal = (min_sal + max_sal) / 2
        return min_sal, max_sal, avg_sal
    except:
        return np.nan, np.nan, np.nan

# Salary Estimate 컬럼을 최소, 최대, 평균 연봉으로 나눠서 저장
df[['min_salary', 'max_salary', 'avg_salary']] = df['Salary Estimate'].apply(lambda x: pd.Series(extract_salary(x)))

# 3. Size 전처리 함수 정의
def map_size(size_str):
    if pd.isnull(size_str) or size_str in ['-1', 'Unknown']:
        return np.nan
    size_str = size_str.strip()
    # 기업 규모를 범주형 그룹으로 매핑
    if size_str in ['0 to 50 employees', '51 to 200 employees']:
        return 'Small'
    elif size_str in ['201 to 500 employees', '501 to 1000 employees']:
        return 'Medium'
    elif size_str in ['1001 to 5000 employees', '5001 to 10000 employees']:
        return 'Large'
    elif size_str == '10000+ employees':
        return 'Very Large'
    return np.nan

# Size_cleaned 컬럼 생성
df['Size_cleaned'] = df['Size'].apply(map_size)

# 4. Founded → Company_age 생성
df['Founded'] = pd.to_numeric(df['Founded'], errors='coerce').replace(-1, np.nan)  # 숫자 변환 및 -1 제거
df['Founded'] = df.groupby('Company Name')['Founded'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))  # 동일 회사 기준 결측값 채우기
df['Founded_cleaned'] = df['Founded'].apply(lambda x: x if pd.notnull(x) and x > 1800 else np.nan)  # 1800년 이상인 경우만 유효
df['Company_age'] = df['Founded_cleaned'].apply(lambda x: 2025 - x if pd.notnull(x) else np.nan)  # 회사 나이 계산

# 5. Job Title → 도메인/직급 인코딩
# 키워드 정의
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
    'Leader': ['leader', 'lead'],
    'Manager': ['manager'],
    'Senior': ['senior', 'sr.', 'sr ', 'sr'],
    'Junior': ['junior', 'jr.', 'jr ', 'associate', 'assistant', 'trainee', 'intern', 'entry'],
    'Staff': ['staff']
}

# 도메인 분류 함수
def classify_domain(title):
    title = str(title).lower()
    for domain, keywords in domain_keywords.items():
        if any(k in title for k in keywords):
            return domain
    return 'Others'

# 직급 분류 함수
def classify_seniority(title):
    title = str(title).lower()
    for level, keywords in seniority_keywords.items():
        if any(k in title for k in keywords):
            return level
    return 'MidLevel'

# 직무 제목을 도메인-직급 형식으로 변환
df['Job Title'] = df['Job Title'].apply(lambda x: f"{classify_domain(x)} - {classify_seniority(x)}")
df[['Job Title', 'Position']] = df['Job Title'].str.split(' - ', expand=True)  # 분리 저장

# 직급 수치형 인코딩
ordinal_map = {'Junior': 1, 'MidLevel': 2, 'Staff': 3, 'Senior': 4, 'Manager': 5, 'Leader': 6, 'Principal': 7}
df['Position_Encoded'] = df['Position'].map(ordinal_map)
df = df[df['Position_Encoded'].notna()]  # 인코딩 실패 행 제거
df['Position_Encoded'] = df['Position_Encoded'].astype(int)

# 6. 그룹별 평균값 계산
group_salary = df.groupby('Size_cleaned')[['min_salary', 'max_salary', 'avg_salary']].mean()
company_age_mean = df.groupby('Company Name')['Company_age'].mean()

# 7. 결측치 보완 함수
def fill_features(row):
    size = row['Size_cleaned']
    company = row['Company Name']
    # 규모 기준 연봉 보완
    if pd.isnull(row['min_salary']) and size in group_salary.index:
        row['min_salary'] = group_salary.loc[size, 'min_salary']
    if pd.isnull(row['max_salary']) and size in group_salary.index:
        row['max_salary'] = group_salary.loc[size, 'max_salary']
    if pd.isnull(row['avg_salary']) and size in group_salary.index:
        row['avg_salary'] = group_salary.loc[size, 'avg_salary']
    # 회사 기준 나이 보완
    if pd.isnull(row['Company_age']) and company in company_age_mean.index:
        row['Company_age'] = company_age_mean[company]
    return row

df = df.apply(fill_features, axis=1)  # 모든 행에 적용

# 8. 남은 결측치 평균/최빈값으로 보완
df['min_salary'] = df['min_salary'].fillna(round(df['min_salary'].mean()))
df['max_salary'] = df['max_salary'].fillna(round(df['max_salary'].mean()))
df['avg_salary'] = df['avg_salary'].fillna(round(df['avg_salary'].mean()))
df['Company_age'] = df['Company_age'].fillna(round(df['Company_age'].mean()))
df['Founded_cleaned'] = df['Founded_cleaned'].fillna(round(df['Founded_cleaned'].mean()))
df['Size_cleaned'] = df['Size_cleaned'].fillna(df['Size_cleaned'].mode()[0])
df['Size'] = df['Size'].fillna(df['Size'].mode()[0])

# 9. Salary Estimate 컬럼 재구성
df['Salary Estimate'] = df.apply(lambda row: f"${int(row['min_salary'])}K-${int(row['max_salary'])}K (Glassdoor est.)", axis=1)

# 10. 원본 Founded 제거
df.drop(columns=['Founded'], inplace=True)

# 11. Location 및 Headquarters → 주(state) 파생 및 그룹화
df['Location'] = df['Location'].str.lower().str.replace(', ', ',', regex=False).str.replace(',', ', ', regex=False)
df['Headquarters'] = df['Headquarters'].str.lower().str.replace(', ', ',', regex=False).str.replace(',', ', ', regex=False)

# 주(state) 추출
df['Location_state'] = df['Location'].apply(lambda x: x.split(',')[1].strip() if pd.notnull(x) and ',' in x else 'Unknown')
df['Headquarters_state'] = df['Headquarters'].apply(lambda x: x.split(',')[1].strip() if pd.notnull(x) and ',' in x else 'Unknown')

# 상위 6개 지역으로 그룹화
top6_location_states = df['Location_state'].value_counts().nlargest(6).index.tolist()
top6_headq_states = df['Headquarters_state'].value_counts().nlargest(6).index.tolist()
df['Location_state_binned'] = df['Location_state'].apply(lambda x: x if x in top6_location_states else 'Other')
df['Headquarters_state_binned'] = df['Headquarters_state'].apply(lambda x: x if x in top6_headq_states else 'Other')

# 근무지가 본사인지 여부
df['works_at_headquarters'] = df.apply(lambda row: 'Yes' if row['Location'] == row['Headquarters'] else 'No', axis=1)

# 12. Ownership 타입 그룹화
most_common_ownership = df['Type of ownership'].mode()[0]
def classify_ownership(val):
    if val in ["-1", "Unknown", "Other Organization"]:
        val = most_common_ownership
    if val in ["Company - Private", "Subsidiary or Business Segment", "Contract", "Private Practice / Firm"]:
        return "Private"
    elif val == "Company - Public":
        return "Public"
    elif val in ["Nonprofit Organization", "Government", "Hospital", "College / University", "School / School District"]:
        return "Public Service"
    return "Other"

df['Ownership_Grouped'] = df['Type of ownership'].apply(classify_ownership)

# 13. 최종 저장 및 결측치 확인
df.to_csv("glassdoor_cleaned.csv", index=False)  # 전처리 완료된 데이터 저장
print("✅ 전처리 완료 및 저장: glassdoor_cleaned.csv")
print("📊 남은 결측치 수:\n", df.isnull().sum())  # 남은 결측치 수 출력


