# Week 5 Platform — Design Spec

**Date:** 2026-04-14  
**Status:** Approved

---

## Overview

Week 5 딥러닝 강의 콘텐츠(Regularization, Overfitting, Data Augmentation, Transfer Learning, CNN-MNIST)를 제공하는 교육 플랫폼. 과제 제출용 사이트로, Google OAuth 로그인과 Polar.sh 결제를 체험할 수 있는 데모 모드를 포함한다.

---

## Tech Stack

| 영역 | 기술 |
|------|------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Auth | NextAuth v5 |
| DB | Turso (hosted SQLite, libsql) |
| Payment | Polar.sh (sandbox) |
| Deploy | Vercel |
| Location | `week5-platform/` (AIandMLcourse 레포 서브폴더) |

---

## 1. 페이지 구조 & 라우팅

```
week5-platform/
├── app/
│   ├── page.tsx                        # 랜딩 페이지
│   ├── chapters/
│   │   ├── page.tsx                    # 챕터 목록
│   │   └── [slug]/page.tsx             # 챕터 상세
│   ├── api/
│   │   ├── auth/[...nextauth]/route.ts # NextAuth 핸들러
│   │   ├── demo-session/route.ts       # 데모 세션 생성
│   │   └── polar/webhook/route.ts      # Polar.sh 웹훅
│   └── layout.tsx
├── components/
│   ├── ChapterCard.tsx
│   ├── DemoButton.tsx
│   ├── PaymentButton.tsx
│   └── ChapterContent.tsx
├── lib/
│   ├── auth.ts                         # NextAuth 설정
│   ├── db.ts                           # Turso 클라이언트
│   └── chapters.ts                     # 챕터 메타데이터
└── .env.local
```

### 라우트별 접근 권한

| 페이지 | 접근 | 설명 |
|--------|------|------|
| `/` | 누구나 | 코스 소개, 로그인/데모 버튼 |
| `/chapters` | 누구나 | 챕터 목록, 유료 챕터는 자물쇠 표시 |
| `/chapters/regularization` | 누구나 | 무료 챕터 |
| `/chapters/overfitting` | 누구나 | 무료 챕터 |
| `/chapters/data-augmentation` | 로그인 + 유료 | 결제 필요 |
| `/chapters/transfer-learning` | 로그인 + 유료 | 결제 필요 |
| `/chapters/cnn-mnist` | 로그인 + 유료 | 결제 필요 |

---

## 2. DB 스키마 (Turso / SQLite)

```sql
CREATE TABLE users (
  id         TEXT PRIMARY KEY,  -- NextAuth session user id
  email      TEXT UNIQUE NOT NULL,
  name       TEXT,
  image      TEXT,
  is_demo    INTEGER DEFAULT 0, -- 1 = 데모 유저
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE subscriptions (
  id                    TEXT PRIMARY KEY,
  user_id               TEXT NOT NULL REFERENCES users(id),
  polar_subscription_id TEXT,
  status                TEXT NOT NULL, -- 'active' | 'cancelled'
  created_at            TEXT DEFAULT (datetime('now'))
);
```

### 접근 로직

- 유료 챕터 요청 시: `subscriptions.status = 'active'` 확인
- 데모 유저(`is_demo = 1`): 유료 챕터 포함 전체 접근 허용
- Polar.sh 웹훅 수신 시: `subscriptions` 테이블 upsert

---

## 3. 인증 & 데모 모드

### Provider 구성 (NextAuth v5)

```
Google OAuth Provider  →  실제 Google 계정 로그인
Credentials Provider   →  데모 로그인 (mock 세션)
```

### 데모 로그인 동작

- "데모로 체험하기" 버튼 클릭
- `demo@example.com` 고정 계정으로 즉시 세션 생성
- DB에 `is_demo = 1` 로 저장
- 유료 챕터 포함 전체 접근 허용
- 세션 만료: 1시간

### Polar.sh 결제 흐름

```
챕터 목록 → 자물쇠/구독 버튼 클릭
→ Polar.sh 샌드박스 체크아웃 (테스트 카드: 4242 4242 4242 4242)
→ 결제 완료 → POST /api/polar/webhook
→ subscriptions 테이블 업데이트 (status = 'active')
→ 유료 챕터 접근 허용
```

---

## 4. UI 컴포넌트

### 랜딩 페이지 (`/`)

- Hero 섹션: 제목 "Week 5 딥러닝 핵심 개념" + 한 줄 설명
- 챕터 미리보기 카드 5개 (무료/유료 배지)
- CTA 버튼: `[Google로 로그인]` / `[데모로 체험하기 →]`

### 챕터 목록 (`/chapters`)

- 2열 카드 그리드
- 무료 챕터: 초록 "무료" 배지
- 유료 챕터: 자물쇠 아이콘 + 회색 처리
- 하단: `[구독하기]` 버튼 (Polar.sh 체크아웃 연결)

### 챕터 상세 (`/chapters/[slug]`)

- 제목 + 개념 설명 (week5.md 내용 기반)
- 결과 이미지 (`week5/outputs/` 폴더)
- 코드 블록 (syntax highlighting)
- 이전/다음 챕터 네비게이션

---

## 5. 환경변수

```env
# NextAuth
NEXTAUTH_SECRET=
NEXTAUTH_URL=

# Google OAuth
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

# Polar.sh (sandbox)
POLAR_PRODUCT_ID=
POLAR_ACCESS_TOKEN=
POLAR_WEBHOOK_SECRET=

# Turso
TURSO_DATABASE_URL=
TURSO_AUTH_TOKEN=

# Config
NEXT_PUBLIC_FREE_USAGE_LIMIT=5
```

---

## 6. 챕터 메타데이터

```ts
const chapters = [
  { slug: 'regularization',     title: '1. Regularization',     free: true  },
  { slug: 'overfitting',        title: '2. Overfitting',         free: true  },
  { slug: 'data-augmentation',  title: '3. Data Augmentation',   free: false },
  { slug: 'transfer-learning',  title: '4. Transfer Learning',   free: false },
  { slug: 'cnn-mnist',          title: '5. CNN - MNIST',         free: false },
]
```
