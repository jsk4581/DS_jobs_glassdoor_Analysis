import pandas as pd
import numpy as np
import re

# ─────────────────────────────────────────────
# Step 1: 데이터 로드
# ─────────────────────────────────────────────
df = pd.read_csv('glassdoor_jobs.csv')  # 원본 데이터 사용

# ─────────────────────────────────────────────
# Step 2: 직무 도메인 분류 및 직급 분류
# ─────────────────────────────────────────────
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
    'Leader': ['leader', 'lead'],  # 수정 사항 반영
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

# ─────────────────────────────────────────────
# Step 3: Job Title 열 분리
# ─────────────────────────────────────────────
split_cols = df['Job Title'].str.split(' - ', expand=True)
df['Job Title'] = split_cols[0]
df['Position'] = split_cols[1]

# ─────────────────────────────────────────────
# Step 4: 오디널 인코딩 (직급)
# ─────────────────────────────────────────────
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
# Step 5: Revenue 전처리
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
# Step 6: 결과 저장
# ─────────────────────────────────────────────
df.to_csv('glassdoor_final_processed.csv', index=False)
print(df[['Job Title', 'Position', 'Position_Encoded', 'Revenue']].head(10))
