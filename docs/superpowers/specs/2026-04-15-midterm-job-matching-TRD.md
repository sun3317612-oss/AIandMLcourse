# TRD: 한국 취업 매칭 플랫폼

**작성일**: 2026-04-15  
**버전**: v1.0  
**연관 문서**: `2026-04-15-midterm-job-matching-PRD.md`

---

## 1. 아키텍처 개요

```
[사용자 브라우저]
       ↓
[Next.js 15 — Vercel]
 - App Router
 - Google OAuth (NextAuth v5)
 - Turso DB 클라이언트 (libsql)
 - shadcn/ui + Tailwind CSS
       ↓ REST API 호출
[FastAPI — Railway]
 - 크롤링 모듈 (스펙업, 링커리어, 사람인)
 - PyTorch MLP 매칭 모델 서빙
 - 합격자 스펙 데이터 관리
       ↓
[Turso DB (SQLite)]     [model.pt]
 - 사용자 / 스펙 데이터    - 오프라인 훈련 후 저장
 - 회사 / 합격 스펙        - FastAPI 시작 시 로드
 - 자소서 문항
```

---

## 2. 기술 스택

| 영역 | 기술 | 비고 |
|------|------|------|
| Frontend | Next.js 15 (App Router) | React 기반 |
| UI 컴포넌트 | shadcn/ui | Tailwind 기반, 전문적 디자인 |
| 스타일 | Tailwind CSS | shadcn/ui 내부 사용 |
| 인증 | NextAuth v5 (Google OAuth) | week5 코드 재활용 |
| DB | Turso (SQLite) | week5 코드 재활용 |
| Backend | FastAPI (Python 3.11+) | |
| ML | PyTorch (MLP) | TensorFlow도 가능 |
| 크롤링 | BeautifulSoup4 + httpx | |
| 외부 API | 사람인 Open API | 자소서 문항 수집 |
| 결제 | Polar.sh | 버튼/UI만, 실제 결제 미활성 |
| 배포 (FE) | Vercel | GitHub 연동 자동 배포 |
| 배포 (BE) | Railway | FastAPI + 모델 파일 |

---

## 3. DB 스키마 (Turso)

```sql
-- 사용자
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용자 스펙
CREATE TABLE user_specs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL REFERENCES users(id),
  gpa REAL,                        -- 학점 (소수점)
  gpa_scale REAL DEFAULT 4.5,      -- 4.3 or 4.5
  university TEXT,                 -- 대학명
  university_tier INTEGER,         -- 1~5 (수동 정의)
  department TEXT,                 -- 학과명
  department_type TEXT,            -- 이공계/상경계/인문계/기타
  toeic INTEGER,                   -- 토익 점수
  opic_grade TEXT,                 -- AL/IH/IM/IL
  toefl INTEGER,                   -- 토플 점수
  certificates TEXT,               -- JSON 배열 ["정보처리기사", ...]
  internship_months INTEGER DEFAULT 0,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 회사 정보
CREATE TABLE companies (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  industry TEXT,                   -- IT/반도체/금융/에너지/...
  size TEXT,                       -- 대기업/중견/스타트업
  logo_url TEXT,
  career_url TEXT                  -- 채용 페이지 URL
);

-- 합격자 스펙 집계 (크롤링 + 수동)
CREATE TABLE accepted_specs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company_id INTEGER REFERENCES companies(id),
  gpa_avg REAL,
  gpa_min REAL,
  toeic_avg INTEGER,
  toeic_min INTEGER,
  university_tier_avg REAL,
  department_type TEXT,            -- 주로 채용하는 계열
  sample_count INTEGER DEFAULT 0, -- 데이터 표본 수
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 자소서 문항
CREATE TABLE cover_letters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  company_id INTEGER REFERENCES companies(id),
  question TEXT NOT NULL,
  max_length INTEGER,              -- 글자 수 제한
  year INTEGER,                   -- 채용 연도
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. ML 파이프라인

### 4.1 입력 벡터 (사용자 스펙 → 숫자)

```python
def vectorize_user(spec):
    return [
        spec.gpa / spec.gpa_scale,              # 0~1 정규화
        UNIVERSITY_TIER[spec.university] / 5,   # 0~1
        DEPT_TYPE_MAP[spec.department_type],    # 이공계=1.0, 상경=0.8, ...
        min(spec.toeic, 990) / 990,             # 0~1
        OPIC_MAP.get(spec.opic_grade, 0),       # 0~1
        min(spec.internship_months, 24) / 24,   # 0~1 (최대 2년)
        len(spec.certificates) / 5,             # 0~1 (최대 5개)
    ]  # 총 7차원 벡터
```

### 4.2 대학 Tier 수동 정의

```python
UNIVERSITY_TIER = {
    # Tier 1
    "서울대학교": 5, "연세대학교": 5, "고려대학교": 5,
    # Tier 2
    "성균관대학교": 4, "한양대학교": 4, "서강대학교": 4, "이화여자대학교": 4,
    # Tier 3
    "중앙대학교": 3, "경희대학교": 3, "한국외국어대학교": 3, "시립대학교": 3,
    # Tier 4
    "건국대학교": 2, "동국대학교": 2, "홍익대학교": 2,
    # Tier 5 (기타)
    "default": 1
}
```

### 4.3 합성 데이터 생성

```python
def generate_synthetic_data(accepted_specs, n_samples=500):
    """
    합격자 스펙 평균값 주변으로 가우시안 노이즈 추가하여
    합격(1) / 불합격(0) 샘플 생성
    """
    data = []
    for company in accepted_specs:
        # 합격 샘플: 기준값 근처 (노이즈 작게)
        for _ in range(n_samples // 2):
            sample = add_noise(company.avg_vector, sigma=0.05)
            data.append((sample, company.id, 1))
        # 불합격 샘플: 기준값 아래 (노이즈 크게)
        for _ in range(n_samples // 2):
            sample = subtract_and_noise(company.avg_vector, sigma=0.1)
            data.append((sample, company.id, 0))
    return data
```

### 4.4 PyTorch MLP 모델

```python
import torch
import torch.nn as nn

class JobMatchingMLP(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # 매칭 점수 0~1
        )

    def forward(self, x):
        return self.network(x)

# 훈련 후 저장
torch.save(model.state_dict(), "model.pt")
```

### 4.5 FastAPI 서빙

```python
# FastAPI 시작 시 모델 로드
model = JobMatchingMLP()
model.load_state_dict(torch.load("model.pt"))
model.eval()

@app.post("/match")
def match_companies(user_spec: UserSpec):
    vec = vectorize_user(user_spec)
    tensor = torch.tensor(vec, dtype=torch.float32)
    
    results = []
    for company in companies:
        score = model(tensor).item()
        results.append({"company": company.name, "score": score})
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:10]
```

---

## 5. API 명세

### Frontend → FastAPI

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/match` | 사용자 스펙 → 회사 매칭 점수 반환 |
| GET | `/companies` | 전체 회사 목록 |
| GET | `/companies/{id}/cover-letters` | 자소서 문항 조회 |
| POST | `/crawl/trigger` | 크롤링 수동 트리거 (관리용) |

### 환경변수

```bash
# Next.js (Vercel)
NEXTAUTH_URL=https://your-app.vercel.app
NEXTAUTH_SECRET=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
TURSO_URL=...
TURSO_AUTH_TOKEN=...
NEXT_PUBLIC_API_URL=https://your-api.railway.app

# FastAPI (Railway)
DATABASE_URL=...  # Turso 연결
SARAMIN_API_KEY=... # 사람인 Open API
```

---

## 6. 크롤링 전략

| 데이터 | 소스 | 방식 | 주기 |
|--------|------|------|------|
| 합격자 스펙 | 스펙업, 링커리어 | BeautifulSoup 크롤링 | 초기 1회 |
| 자소서 문항 | 사람인 API | Open API | 초기 1회 |
| 회사 채용 URL | 각 회사 채용 페이지 | 수동 입력 | 수동 |

> 과제 제출용 기준: 크롤링은 초기 데이터셋 구축 시 1회 실행, 이후 DB에 저장된 데이터 사용

---

## 7. 프로젝트 디렉토리 구조

```
midterm-project/
├── frontend/                  # Next.js
│   ├── app/
│   │   ├── page.tsx           # 랜딩
│   │   ├── profile/page.tsx   # 스펙 입력
│   │   ├── results/page.tsx   # 매칭 결과
│   │   └── me/page.tsx        # 마이페이지
│   ├── components/
│   │   ├── ui/                # shadcn/ui 컴포넌트
│   │   ├── SpecForm.tsx       # 스펙 입력 폼
│   │   ├── CompanyCard.tsx    # 회사 카드
│   │   └── ScoreBar.tsx       # 점수 바
│   └── lib/
│       ├── auth.ts            # NextAuth 설정
│       └── db.ts              # Turso 클라이언트
│
├── backend/                   # FastAPI
│   ├── main.py
│   ├── routers/
│   │   ├── match.py           # 매칭 API
│   │   └── companies.py       # 회사 API
│   ├── ml/
│   │   ├── model.py           # PyTorch MLP 정의
│   │   ├── train.py           # 훈련 스크립트
│   │   ├── vectorize.py       # 스펙 벡터화
│   │   └── model.pt           # 훈련된 모델 (git 제외)
│   └── crawlers/
│       ├── specup.py
│       └── linkareer.py
│
└── data/
    └── universities.json      # 대학 Tier 수동 정의
```

---

## 8. 배포 순서

1. Railway에 FastAPI 배포 → URL 확인
2. Vercel 환경변수에 `NEXT_PUBLIC_API_URL` 설정
3. Vercel에 Next.js 배포
4. 크롤링 1회 실행 → DB 데이터 확인
5. PyTorch 모델 훈련 → `model.pt` Railway에 업로드
6. 전체 플로우 E2E 테스트

---

## 9. 타임라인 (2주)

| 기간 | 작업 |
|------|------|
| Day 1~2 | 프로젝트 세팅, DB 스키마, Google OAuth |
| Day 3~4 | FastAPI 기본 구조 + 크롤링 모듈 |
| Day 5~6 | PyTorch 모델 훈련 + 매칭 API |
| Day 7~8 | Frontend 랜딩 + 스펙 입력 페이지 |
| Day 9~10 | 매칭 결과 페이지 + 자소서 모달 |
| Day 11~12 | 마이페이지 + Polar.sh 버튼 |
| Day 13~14 | 전체 테스트, 배포, 영상 촬영 |
