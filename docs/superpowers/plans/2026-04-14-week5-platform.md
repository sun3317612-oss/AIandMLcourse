# Week 5 Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `week5-platform/` 서브폴더에 Next.js 14 기반 딥러닝 교육 플랫폼 구축 — Google OAuth + 데모 로그인, Polar.sh 샌드박스 결제, Vercel 배포.

**Architecture:** Next.js 14 App Router. 무료 챕터(1-2)는 모두 접근 가능, 유료 챕터(3-5)는 `subscriptions.status = 'active'` 또는 `is_demo = 1` 확인 후 허용. Turso(hosted SQLite)로 users/subscriptions 관리. NextAuth v5는 Google OAuth + Credentials(데모) 두 Provider 병행.

**Tech Stack:** Next.js 14, Tailwind CSS v3, NextAuth v5 (beta), @libsql/client, @polar-sh/sdk, Vercel

---

## File Map

```
week5-platform/
├── app/
│   ├── layout.tsx                          # Root layout + SessionProvider
│   ├── page.tsx                            # 랜딩 페이지
│   ├── chapters/
│   │   ├── page.tsx                        # 챕터 목록
│   │   └── [slug]/
│   │       └── page.tsx                    # 챕터 상세
│   └── api/
│       ├── auth/[...nextauth]/route.ts     # NextAuth 핸들러
│       ├── demo-login/route.ts             # 데모 세션 생성 (POST)
│       └── polar/
│           ├── checkout/route.ts           # Polar 체크아웃 URL 생성
│           └── webhook/route.ts            # 결제 완료 웹훅
├── components/
│   ├── Header.tsx                          # 네비게이션 + 로그인 상태
│   ├── ChapterCard.tsx                     # 챕터 카드 (무료/잠금)
│   ├── DemoButton.tsx                      # 데모 로그인 버튼
│   ├── PaymentButton.tsx                   # 구독하기 버튼
│   └── ChapterContent.tsx                  # 마크다운 렌더러
├── lib/
│   ├── auth.ts                             # NextAuth v5 설정 (handlers, auth, signIn, signOut)
│   ├── db.ts                               # Turso 클라이언트 + 쿼리 함수
│   └── chapters.ts                         # 챕터 메타데이터 + 콘텐츠
├── types/
│   └── next-auth.d.ts                      # Session 타입 확장
├── middleware.ts                           # 유료 챕터 접근 제어
├── next.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── .env.local
```

---

## Task 1: 프로젝트 초기화

**Files:**
- Create: `week5-platform/` (전체 Next.js 프로젝트)
- Create: `week5-platform/.env.local`
- Create: `week5-platform/public/images/` (week5 결과 이미지)

- [ ] **Step 1: Next.js 앱 생성**

레포 루트(`AIandMLcourse/`)에서 실행:
```bash
npx create-next-app@latest week5-platform \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --src-dir no \
  --import-alias "@/*"
cd week5-platform
```

- [ ] **Step 2: 추가 패키지 설치**

```bash
npm install next-auth@beta @libsql/client @polar-sh/sdk
npm install react-markdown remark-gfm
npm install -D @types/node
```

- [ ] **Step 3: week5 결과 이미지 복사**

`week5-platform/` 디렉토리에서:
```bash
mkdir -p public/images/week5
cp ../week5/outputs/01_regularization_plot.png public/images/week5/
cp ../week5/outputs/02_overfitting_underfitting.png public/images/week5/
cp ../week5/outputs/03_augmentation_examples.png public/images/week5/
cp ../week5/outputs/05_mnist_cnn_result.png public/images/week5/
```

- [ ] **Step 4: .env.local 작성**

`week5-platform/.env.local`:
```env
# NextAuth
NEXTAUTH_SECRET=mysecretkey123abc
NEXTAUTH_URL=http://localhost:3000

# Google OAuth (Google Cloud Console에서 발급)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Polar.sh Sandbox
POLAR_PRODUCT_ID=your_polar_product_id
POLAR_ACCESS_TOKEN=your_polar_access_token
POLAR_WEBHOOK_SECRET=your_polar_webhook_secret

# Turso (Task 2에서 발급)
TURSO_DATABASE_URL=libsql://your-db-name.turso.io
TURSO_AUTH_TOKEN=your_turso_auth_token

# Config
NEXT_PUBLIC_FREE_USAGE_LIMIT=5
```

- [ ] **Step 5: 기본 실행 확인**

```bash
npm run dev
```
Expected: `http://localhost:3000` 에서 기본 Next.js 페이지 확인

- [ ] **Step 6: 커밋**

```bash
cd ..
git add week5-platform/
git commit -m "feat: week5-platform Next.js 프로젝트 초기화"
```

---

## Task 2: Turso DB 셋업 + 쿼리 함수

**Files:**
- Create: `week5-platform/lib/db.ts`

### 사전 작업: Turso CLI로 DB 생성

터미널에서 직접 실행 (Claude가 실행 불가):
```bash
# Turso CLI 설치 (미설치 시)
npm install -g @tursodatabase/cli

# 로그인
turso auth login

# DB 생성
turso db create week5-platform

# URL과 토큰 확인
turso db show week5-platform --url
turso db tokens create week5-platform
```
→ 출력된 URL과 토큰을 `.env.local`의 `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`에 입력

- [ ] **Step 1: lib/db.ts 작성**

`week5-platform/lib/db.ts`:
```ts
import { createClient } from "@libsql/client"

const db = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_AUTH_TOKEN!,
})

export async function initDB() {
  await db.executeMultiple(`
    CREATE TABLE IF NOT EXISTS users (
      id         TEXT PRIMARY KEY,
      email      TEXT UNIQUE NOT NULL,
      name       TEXT,
      image      TEXT,
      is_demo    INTEGER DEFAULT 0,
      created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS subscriptions (
      id                    TEXT PRIMARY KEY,
      user_id               TEXT NOT NULL REFERENCES users(id),
      polar_subscription_id TEXT,
      status                TEXT NOT NULL,
      created_at            TEXT DEFAULT (datetime('now'))
    );
  `)
}

export async function upsertUser(user: {
  id: string
  email: string
  name?: string | null
  image?: string | null
  isDemo?: boolean
}) {
  await db.execute({
    sql: `INSERT INTO users (id, email, name, image, is_demo)
          VALUES (:id, :email, :name, :image, :is_demo)
          ON CONFLICT(id) DO UPDATE SET name = :name, image = :image`,
    args: {
      id: user.id,
      email: user.email,
      name: user.name ?? null,
      image: user.image ?? null,
      is_demo: user.isDemo ? 1 : 0,
    },
  })
}

export async function getUser(id: string) {
  const result = await db.execute({
    sql: `SELECT * FROM users WHERE id = ?`,
    args: [id],
  })
  return result.rows[0] ?? null
}

export async function hasActiveSubscription(userId: string): Promise<boolean> {
  const result = await db.execute({
    sql: `SELECT id FROM subscriptions WHERE user_id = ? AND status = 'active' LIMIT 1`,
    args: [userId],
  })
  return result.rows.length > 0
}

export async function upsertSubscription(sub: {
  id: string
  userId: string
  polarSubscriptionId: string
  status: "active" | "cancelled"
}) {
  await db.execute({
    sql: `INSERT INTO subscriptions (id, user_id, polar_subscription_id, status)
          VALUES (:id, :user_id, :polar_id, :status)
          ON CONFLICT(id) DO UPDATE SET status = :status`,
    args: {
      id: sub.id,
      user_id: sub.userId,
      polar_id: sub.polarSubscriptionId,
      status: sub.status,
    },
  })
}

export { db }
```

- [ ] **Step 2: DB 초기화 스크립트 실행 확인**

`week5-platform/` 에서:
```bash
node -e "
const { createClient } = require('@libsql/client');
require('dotenv').config({ path: '.env.local' });
const db = createClient({
  url: process.env.TURSO_DATABASE_URL,
  authToken: process.env.TURSO_AUTH_TOKEN,
});
db.executeMultiple(\`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
    name TEXT, image TEXT, is_demo INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS subscriptions (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    polar_subscription_id TEXT, status TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
  );
\`).then(() => console.log('DB init OK')).catch(console.error);
"
```
Expected: `DB init OK`

- [ ] **Step 3: 커밋**

```bash
git add week5-platform/lib/db.ts
git commit -m "feat: Turso DB 클라이언트 + users/subscriptions 스키마"
```

---

## Task 3: 챕터 데이터 모듈

**Files:**
- Create: `week5-platform/lib/chapters.ts`
- Create: `week5-platform/types/next-auth.d.ts`

- [ ] **Step 1: lib/chapters.ts 작성**

`week5-platform/lib/chapters.ts`:
```ts
export type Chapter = {
  slug: string
  title: string
  description: string
  free: boolean
  image?: string
  content: string
}

export const chapters: Chapter[] = [
  {
    slug: "regularization",
    title: "1. Regularization (규제)",
    description: "L1/L2, Dropout, Batch Normalization으로 과적합을 막는 기법",
    free: true,
    image: "/images/week5/01_regularization_plot.png",
    content: `
## 개념

모델이 훈련 데이터에 과도하게 맞춰지는 **과적합(Overfitting)**을 막기 위한 기법들입니다.

### L1 / L2 Regularization

가중치(Weight)의 크기에 페널티를 부여하여 모델을 단순하게 만듭니다.

- **L1**: 가중치의 절대값 합을 손실 함수에 추가 → 일부 가중치를 0으로 만들어 희소성(Sparsity) 확보
- **L2**: 가중치의 제곱 합을 손실 함수에 추가 → 가중치를 전반적으로 작게 유지

### Dropout

학습 시 무작위로 일부 뉴런을 꺼버려서(0으로 만들어) 특정 뉴런에 대한 의존을 방지합니다.

### Batch Normalization

각 층의 입력을 평균 0, 분산 1로 정규화하여 학습을 안정화하고 속도를 높입니다.

\`\`\`python
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(1)
])
\`\`\`

## 결과 해석

- **None**: 규제 없이 훈련하면 Validation Loss가 다시 증가 (과적합)
- **Dropout/L2**: 과적합 억제 → Validation Loss가 안정적으로 유지
    `,
  },
  {
    slug: "overfitting",
    title: "2. Overfitting vs Underfitting",
    description: "모델 복잡도와 일반화 성능의 관계를 시각적으로 이해",
    free: true,
    image: "/images/week5/02_overfitting_underfitting.png",
    content: `
## 개념

| 상태 | 설명 |
|------|------|
| **Underfitting** | 모델이 너무 단순 → 데이터 패턴을 제대로 학습 못함 |
| **Overfitting** | 모델이 너무 복잡 → 훈련 데이터의 노이즈까지 학습 |
| **Balanced** | 적절한 복잡도 → 일반화 성능 우수 |

## Loss Curve로 진단하기

\`\`\`python
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=100)

# Train Loss vs Validation Loss 비교
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
\`\`\`

## 결과 해석

- **Overfit 모델**: Train Loss ↓ 낮음, Val Loss ↑ 높음 (격차 큼)
- **Underfit 모델**: Train Loss도 높고, Val Loss도 높음
- **Balanced 모델**: 두 Loss 모두 수렴하며 격차 작음
    `,
  },
  {
    slug: "data-augmentation",
    title: "3. Data Augmentation (데이터 증강)",
    description: "데이터 부족 문제를 해결하는 이미지 증강 기법",
    free: false,
    image: "/images/week5/03_augmentation_examples.png",
    content: `
## 개념

데이터가 부족할 때, 기존 이미지를 변형(회전, 뒤집기, 확대/축소 등)하여 데이터의 다양성을 늘립니다.
모델이 위치나 각도 변화에 강건(Robust)해지도록 돕습니다.

## Keras로 구현

\`\`\`python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
])

# 모델 파이프라인에 통합
model = tf.keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax'),
])
\`\`\`

## 언제 사용하나?

- 훈련 데이터가 수백~수천 장 이하일 때
- 이미지 분류, 객체 탐지 등 컴퓨터 비전 태스크
- Transfer Learning과 함께 사용 시 효과 극대화
    `,
  },
  {
    slug: "transfer-learning",
    title: "4. Transfer Learning (전이 학습)",
    description: "ImageNet 사전 학습 모델을 활용해 적은 데이터로 높은 성능 달성",
    free: false,
    image: undefined,
    content: `
## 개념

이미 대량의 데이터(ImageNet 1.2M장)로 학습된 모델의 지식을 가져와,
내가 가진 적은 데이터의 문제 해결에 활용하는 방법입니다.

| 방법 | 설명 |
|------|------|
| **Feature Extraction** | Pre-trained 모델의 Conv Base를 고정(Freeze), 분류기만 새로 학습 |
| **Fine-tuning** | Pre-trained 모델의 상위 층 일부도 미세하게 같이 학습 |

## MobileNetV2로 구현

\`\`\`python
# 사전 학습된 모델 불러오기 (분류기 제외)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)

# Feature Extraction: Base 동결
base_model.trainable = False

# 새 분류기 추가
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])

print(f"학습 가능 파라미터: {model.trainable_variables.__len__()}")
\`\`\`

## 핵심 인사이트

- ImageNet의 저수준 특징(에지, 텍스처)은 대부분의 이미지 태스크에 공통으로 유용
- 데이터가 100장뿐이어도 ResNet50 수준의 성능을 낼 수 있는 이유
    `,
  },
  {
    slug: "cnn-mnist",
    title: "5. CNN 실습 — MNIST 손글씨 인식",
    description: "Conv2D + MaxPooling2D로 99% 정확도 달성하는 CNN 구현",
    free: false,
    image: "/images/week5/05_mnist_cnn_result.png",
    content: `
## CNN 구조

**CNN (Convolutional Neural Network)**은 이미지 처리에 특화된 딥러닝 구조입니다.

| 레이어 | 역할 |
|--------|------|
| **Conv2D** | 이미지의 특징(에지, 패턴) 추출 |
| **MaxPooling2D** | 이미지 크기 축소 + 중요 특징 보존 |
| **Flatten** | 2D 특징 맵 → 1D 벡터 변환 |
| **Dense** | 최종 분류 수행 |

## MNIST CNN 구현

\`\`\`python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=10, validation_split=0.1)
\`\`\`

## 결과

- 학습 정확도: **~99%**
- 검증 정확도: **~99%**
- Dense-only 모델 대비 파라미터 수 감소 + 정확도 향상
    `,
  },
]

export function getChapter(slug: string): Chapter | undefined {
  return chapters.find((c) => c.slug === slug)
}

export function getPrevNext(slug: string): {
  prev: Chapter | null
  next: Chapter | null
} {
  const idx = chapters.findIndex((c) => c.slug === slug)
  return {
    prev: idx > 0 ? chapters[idx - 1] : null,
    next: idx < chapters.length - 1 ? chapters[idx + 1] : null,
  }
}
```

- [ ] **Step 2: types/next-auth.d.ts 작성**

`week5-platform/types/next-auth.d.ts`:
```ts
import "next-auth"

declare module "next-auth" {
  interface Session {
    user: {
      id: string
      email: string
      name?: string | null
      image?: string | null
      isDemo: boolean
      isPaid: boolean
    }
  }

  interface User {
    id: string
    email: string
    name?: string | null
    image?: string | null
    isDemo?: boolean
  }
}
```

- [ ] **Step 3: 챕터 함수 테스트**

`week5-platform/` 에서:
```bash
node -e "
const { chapters, getChapter, getPrevNext } = require('./lib/chapters.ts');
" 2>&1 || echo "TypeScript - ts-node로 검증 필요"
```

수동 확인: `chapters.length === 5`, `chapters[0].free === true`, `chapters[2].free === false`

- [ ] **Step 4: 커밋**

```bash
git add week5-platform/lib/chapters.ts week5-platform/types/
git commit -m "feat: 챕터 데이터 모듈 + NextAuth 타입 확장"
```

---

## Task 4: NextAuth v5 설정

**Files:**
- Create: `week5-platform/auth.ts`
- Create: `week5-platform/app/api/auth/[...nextauth]/route.ts`

- [ ] **Step 1: auth.ts 작성**

`week5-platform/auth.ts`:
```ts
import NextAuth from "next-auth"
import Google from "next-auth/providers/google"
import Credentials from "next-auth/providers/credentials"
import { upsertUser, getUser, hasActiveSubscription } from "@/lib/db"

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    Credentials({
      id: "demo",
      name: "Demo",
      credentials: {},
      async authorize() {
        return {
          id: "demo-user-fixed",
          email: "demo@example.com",
          name: "Demo User",
          image: null,
          isDemo: true,
        }
      },
    }),
  ],
  callbacks: {
    async signIn({ user, account }) {
      const isDemo = account?.provider === "demo"
      await upsertUser({
        id: user.id!,
        email: user.email!,
        name: user.name,
        image: user.image,
        isDemo,
      })
      return true
    },
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id
        token.isDemo = (user as { isDemo?: boolean }).isDemo ?? false
      }
      return token
    },
    async session({ session, token }) {
      const dbUser = await getUser(token.id as string)
      const isPaid = dbUser
        ? Boolean(dbUser.is_demo) ||
          (await hasActiveSubscription(token.id as string))
        : false

      session.user.id = token.id as string
      session.user.isDemo = Boolean(token.isDemo)
      session.user.isPaid = isPaid
      return session
    },
  },
  pages: {
    signIn: "/",
  },
  secret: process.env.NEXTAUTH_SECRET,
})
```

- [ ] **Step 2: NextAuth API Route 작성**

`week5-platform/app/api/auth/[...nextauth]/route.ts`:
```ts
import { handlers } from "@/auth"
export const { GET, POST } = handlers
```

- [ ] **Step 3: 커밋**

```bash
git add week5-platform/auth.ts week5-platform/app/api/auth/
git commit -m "feat: NextAuth v5 Google OAuth + 데모 Credentials 설정"
```

---

## Task 5: Layout + Header

**Files:**
- Create: `week5-platform/app/layout.tsx`
- Create: `week5-platform/components/Header.tsx`

- [ ] **Step 1: Header.tsx 작성**

`week5-platform/components/Header.tsx`:
```tsx
"use client"

import Link from "next/link"
import { useSession, signOut } from "next-auth/react"

export default function Header() {
  const { data: session } = useSession()

  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
        <Link href="/" className="text-lg font-bold text-gray-900">
          Week 5 딥러닝
        </Link>
        <nav className="flex items-center gap-4">
          <Link href="/chapters" className="text-sm text-gray-600 hover:text-gray-900">
            챕터 목록
          </Link>
          {session ? (
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-600">
                {session.user.isDemo ? "데모 사용자" : session.user.name}
              </span>
              <button
                onClick={() => signOut({ callbackUrl: "/" })}
                className="rounded-md border border-gray-300 px-3 py-1 text-sm hover:bg-gray-50"
              >
                로그아웃
              </button>
            </div>
          ) : null}
        </nav>
      </div>
    </header>
  )
}
```

- [ ] **Step 2: app/layout.tsx 작성**

`week5-platform/app/layout.tsx`:
```tsx
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { SessionProvider } from "next-auth/react"
import Header from "@/components/Header"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Week 5 딥러닝 핵심 개념",
  description: "Regularization, CNN, Transfer Learning 등 딥러닝 핵심 기법 학습",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body className={inter.className}>
        <SessionProvider>
          <Header />
          <main>{children}</main>
        </SessionProvider>
      </body>
    </html>
  )
}
```

- [ ] **Step 3: 실행 확인**

```bash
npm run dev
```
Expected: `localhost:3000` — Header가 렌더링되고 콘솔 에러 없음

- [ ] **Step 4: 커밋**

```bash
git add week5-platform/app/layout.tsx week5-platform/components/Header.tsx
git commit -m "feat: Layout + Header (SessionProvider 포함)"
```

---

## Task 6: 랜딩 페이지 + 데모 로그인

**Files:**
- Create: `week5-platform/app/page.tsx`
- Create: `week5-platform/components/DemoButton.tsx`
- Create: `week5-platform/app/api/demo-login/route.ts`

- [ ] **Step 1: DemoButton.tsx 작성**

`week5-platform/components/DemoButton.tsx`:
```tsx
"use client"

import { signIn } from "next-auth/react"
import { useState } from "react"

export default function DemoButton() {
  const [loading, setLoading] = useState(false)

  async function handleDemo() {
    setLoading(true)
    await signIn("demo", { callbackUrl: "/chapters" })
  }

  return (
    <button
      onClick={handleDemo}
      disabled={loading}
      className="rounded-lg bg-gray-900 px-6 py-3 text-white font-medium hover:bg-gray-700 disabled:opacity-50 transition-colors"
    >
      {loading ? "로딩 중..." : "데모로 체험하기 →"}
    </button>
  )
}
```

- [ ] **Step 2: app/page.tsx 작성**

`week5-platform/app/page.tsx`:
```tsx
import Link from "next/link"
import { signIn } from "@/auth"
import DemoButton from "@/components/DemoButton"
import { chapters } from "@/lib/chapters"

export default function LandingPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-16">
      {/* Hero */}
      <div className="text-center mb-16">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Week 5 딥러닝 핵심 개념
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Regularization, Overfitting, CNN — 실습 코드와 시각화로 배우는 딥러닝
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <form
            action={async () => {
              "use server"
              await signIn("google", { redirectTo: "/chapters" })
            }}
          >
            <button
              type="submit"
              className="rounded-lg border border-gray-300 bg-white px-6 py-3 font-medium hover:bg-gray-50 transition-colors"
            >
              Google로 로그인
            </button>
          </form>
          <DemoButton />
        </div>
      </div>

      {/* 챕터 미리보기 */}
      <div>
        <h2 className="text-2xl font-semibold text-gray-900 mb-6">커리큘럼</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {chapters.map((chapter) => (
            <div
              key={chapter.slug}
              className="rounded-lg border border-gray-200 p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-gray-900">{chapter.title}</h3>
                {chapter.free ? (
                  <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
                    무료
                  </span>
                ) : (
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-500">
                    유료
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-500">{chapter.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: 브라우저에서 확인**

```bash
npm run dev
```
Expected: 랜딩 페이지에서 "Google로 로그인" + "데모로 체험하기 →" 버튼 표시

- [ ] **Step 4: 커밋**

```bash
git add week5-platform/app/page.tsx week5-platform/components/DemoButton.tsx
git commit -m "feat: 랜딩 페이지 + 데모 로그인 버튼"
```

---

## Task 7: 챕터 목록 페이지

**Files:**
- Create: `week5-platform/components/ChapterCard.tsx`
- Create: `week5-platform/components/PaymentButton.tsx`
- Create: `week5-platform/app/chapters/page.tsx`
- Create: `week5-platform/app/api/polar/checkout/route.ts`

- [ ] **Step 1: PaymentButton.tsx 작성**

`week5-platform/components/PaymentButton.tsx`:
```tsx
"use client"

import { useState } from "react"

export default function PaymentButton() {
  const [loading, setLoading] = useState(false)

  async function handleCheckout() {
    setLoading(true)
    const res = await fetch("/api/polar/checkout", { method: "POST" })
    const { url } = await res.json()
    if (url) window.location.href = url
    else setLoading(false)
  }

  return (
    <button
      onClick={handleCheckout}
      disabled={loading}
      className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50 transition-colors"
    >
      {loading ? "처리 중..." : "구독하기 (Polar.sh)"}
    </button>
  )
}
```

- [ ] **Step 2: Polar Checkout API Route 작성**

`week5-platform/app/api/polar/checkout/route.ts`:
```ts
import { NextResponse } from "next/server"
import { auth } from "@/auth"
import { Polar } from "@polar-sh/sdk"

const polar = new Polar({
  accessToken: process.env.POLAR_ACCESS_TOKEN!,
  server: "sandbox",
})

export async function POST() {
  const session = await auth()
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
  }

  const checkout = await polar.checkouts.create({
    productId: process.env.POLAR_PRODUCT_ID!,
    successUrl: `${process.env.NEXTAUTH_URL}/chapters?success=true`,
    customerEmail: session.user.email,
    metadata: { user_id: session.user.id },
  })

  return NextResponse.json({ url: checkout.url })
}
```

- [ ] **Step 3: ChapterCard.tsx 작성**

`week5-platform/components/ChapterCard.tsx`:
```tsx
import Link from "next/link"
import { Chapter } from "@/lib/chapters"

type Props = {
  chapter: Chapter
  accessible: boolean
}

export default function ChapterCard({ chapter, accessible }: Props) {
  return (
    <div className="rounded-lg border border-gray-200 p-5 hover:border-gray-300 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <h3 className="font-semibold text-gray-900">{chapter.title}</h3>
        {chapter.free ? (
          <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 shrink-0 ml-2">
            무료
          </span>
        ) : (
          <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-500 shrink-0 ml-2">
            {accessible ? "유료 ✓" : "🔒 유료"}
          </span>
        )}
      </div>
      <p className="text-sm text-gray-500 mb-4">{chapter.description}</p>
      {accessible ? (
        <Link
          href={`/chapters/${chapter.slug}`}
          className="inline-block rounded-md bg-gray-900 px-4 py-1.5 text-sm text-white hover:bg-gray-700"
        >
          학습하기 →
        </Link>
      ) : (
        <span className="inline-block rounded-md bg-gray-100 px-4 py-1.5 text-sm text-gray-400 cursor-not-allowed">
          잠금
        </span>
      )}
    </div>
  )
}
```

- [ ] **Step 4: app/chapters/page.tsx 작성**

`week5-platform/app/chapters/page.tsx`:
```tsx
import { auth } from "@/auth"
import { chapters } from "@/lib/chapters"
import ChapterCard from "@/components/ChapterCard"
import PaymentButton from "@/components/PaymentButton"

export default async function ChaptersPage({
  searchParams,
}: {
  searchParams: { success?: string }
}) {
  const session = await auth()
  const isPaid = session?.user?.isPaid ?? false

  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      <h1 className="text-3xl font-bold text-gray-900 mb-2">챕터 목록</h1>
      <p className="text-gray-500 mb-8">
        Week 5 딥러닝 핵심 개념 — 5개 챕터
      </p>

      {searchParams.success && (
        <div className="mb-6 rounded-lg bg-green-50 border border-green-200 p-4 text-green-800">
          결제가 완료되었습니다! 모든 챕터에 접근할 수 있습니다.
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
        {chapters.map((chapter) => (
          <ChapterCard
            key={chapter.slug}
            chapter={chapter}
            accessible={chapter.free || isPaid}
          />
        ))}
      </div>

      {!isPaid && session && !session.user.isDemo && (
        <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-6 text-center">
          <h2 className="text-lg font-semibold text-indigo-900 mb-2">
            유료 챕터 잠금 해제
          </h2>
          <p className="text-sm text-indigo-700 mb-4">
            Data Augmentation, Transfer Learning, CNN-MNIST 챕터에 접근하세요.
          </p>
          <div className="max-w-xs mx-auto">
            <PaymentButton />
          </div>
          <p className="text-xs text-indigo-500 mt-2">
            테스트 카드: 4242 4242 4242 4242
          </p>
        </div>
      )}

      {!session && (
        <div className="rounded-lg border border-gray-200 bg-gray-50 p-6 text-center">
          <p className="text-gray-600 mb-3">로그인 후 유료 챕터에 접근하세요.</p>
          <a href="/" className="text-indigo-600 underline text-sm">
            로그인하러 가기
          </a>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 5: 브라우저 확인**

```bash
npm run dev
```
Expected: `/chapters` 페이지에서 무료 챕터(초록 배지), 유료 챕터(자물쇠) 표시

- [ ] **Step 6: 커밋**

```bash
git add week5-platform/components/ week5-platform/app/chapters/page.tsx week5-platform/app/api/polar/checkout/
git commit -m "feat: 챕터 목록 페이지 + PaymentButton + Polar checkout API"
```

---

## Task 8: 챕터 상세 페이지

**Files:**
- Create: `week5-platform/components/ChapterContent.tsx`
- Create: `week5-platform/app/chapters/[slug]/page.tsx`

- [ ] **Step 1: ChapterContent.tsx 작성**

`week5-platform/components/ChapterContent.tsx`:
```tsx
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import Image from "next/image"

type Props = {
  content: string
  image?: string
  title: string
}

export default function ChapterContent({ content, image, title }: Props) {
  return (
    <article className="prose prose-gray max-w-none">
      <h1>{title}</h1>
      {image && (
        <div className="my-6 rounded-lg overflow-hidden border border-gray-200">
          <Image
            src={image}
            alt={`${title} 결과`}
            width={800}
            height={400}
            className="w-full object-contain"
          />
        </div>
      )}
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </article>
  )
}
```

- [ ] **Step 2: Tailwind prose 플러그인 설치**

```bash
npm install @tailwindcss/typography
```

`week5-platform/tailwind.config.ts` 에 플러그인 추가:
```ts
import type { Config } from "tailwindcss"

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: { extend: {} },
  plugins: [require("@tailwindcss/typography")],
}
export default config
```

- [ ] **Step 3: app/chapters/[slug]/page.tsx 작성**

`week5-platform/app/chapters/[slug]/page.tsx`:
```tsx
import { notFound, redirect } from "next/navigation"
import Link from "next/link"
import { auth } from "@/auth"
import { getChapter, getPrevNext } from "@/lib/chapters"
import ChapterContent from "@/components/ChapterContent"

export default async function ChapterPage({
  params,
}: {
  params: { slug: string }
}) {
  const chapter = getChapter(params.slug)
  if (!chapter) notFound()

  // 유료 챕터 접근 제어
  if (!chapter.free) {
    const session = await auth()
    if (!session) redirect("/")
    if (!session.user.isPaid) redirect("/chapters")
  }

  const { prev, next } = getPrevNext(params.slug)

  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      <Link
        href="/chapters"
        className="text-sm text-gray-500 hover:text-gray-700 mb-6 inline-block"
      >
        ← 챕터 목록으로
      </Link>

      <ChapterContent
        title={chapter.title}
        content={chapter.content}
        image={chapter.image}
      />

      {/* 이전/다음 네비게이션 */}
      <div className="mt-12 flex justify-between border-t border-gray-200 pt-6">
        {prev ? (
          <Link
            href={`/chapters/${prev.slug}`}
            className="text-sm text-indigo-600 hover:underline"
          >
            ← {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next && (
          <Link
            href={`/chapters/${next.slug}`}
            className="text-sm text-indigo-600 hover:underline"
          >
            {next.title} →
          </Link>
        )}
      </div>
    </div>
  )
}

export async function generateStaticParams() {
  const { chapters } = await import("@/lib/chapters")
  return chapters.map((c) => ({ slug: c.slug }))
}
```

- [ ] **Step 4: 브라우저 확인**

```bash
npm run dev
```
Expected:
- `/chapters/regularization` → 내용 렌더링됨 (로그인 불필요)
- `/chapters/data-augmentation` → 로그인 없이 접근 시 `/` 로 리디렉션
- 데모 로그인 후 `/chapters/data-augmentation` → 내용 표시됨

- [ ] **Step 5: 커밋**

```bash
git add week5-platform/components/ChapterContent.tsx week5-platform/app/chapters/[slug]/
git commit -m "feat: 챕터 상세 페이지 + 접근 제어 + 이전/다음 네비게이션"
```

---

## Task 9: Polar.sh 웹훅 처리

**Files:**
- Create: `week5-platform/app/api/polar/webhook/route.ts`

- [ ] **Step 1: webhook/route.ts 작성**

`week5-platform/app/api/polar/webhook/route.ts`:
```ts
import { NextRequest, NextResponse } from "next/server"
import { validateEvent, WebhookVerificationError } from "@polar-sh/sdk/webhooks"
import { upsertSubscription } from "@/lib/db"

export async function POST(req: NextRequest) {
  const body = await req.text()
  const signature = req.headers.get("webhook-signature") ?? ""

  let event: ReturnType<typeof validateEvent>
  try {
    event = validateEvent(body, req.headers, process.env.POLAR_WEBHOOK_SECRET!)
  } catch (e) {
    if (e instanceof WebhookVerificationError) {
      return NextResponse.json({ error: "Invalid signature" }, { status: 403 })
    }
    throw e
  }

  if (
    event.type === "subscription.created" ||
    event.type === "subscription.updated"
  ) {
    const sub = event.data
    const userId = (sub.metadata as Record<string, string>)?.user_id

    if (userId) {
      await upsertSubscription({
        id: sub.id,
        userId,
        polarSubscriptionId: sub.id,
        status: sub.status === "active" ? "active" : "cancelled",
      })
    }
  }

  return NextResponse.json({ received: true })
}
```

- [ ] **Step 2: 웹훅 로컬 테스트 (ngrok)**

Polar.sh 샌드박스 웹훅 테스트를 위해:
```bash
# ngrok으로 로컬 서버 외부 노출
npx ngrok http 3000
```

Polar.sh 대시보드 → Webhooks → URL: `https://<ngrok-url>/api/polar/webhook`

- [ ] **Step 3: 커밋**

```bash
git add week5-platform/app/api/polar/webhook/
git commit -m "feat: Polar.sh 웹훅 핸들러 (구독 상태 DB 반영)"
```

---

## Task 10: Vercel 배포 설정

**Files:**
- Create: `week5-platform/vercel.json`

- [ ] **Step 1: vercel.json 작성**

`week5-platform/vercel.json`:
```json
{
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "outputDirectory": ".next"
}
```

- [ ] **Step 2: next.config.ts 이미지 도메인 설정**

`week5-platform/next.config.ts`:
```ts
import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [],
  },
}

export default nextConfig
```

- [ ] **Step 3: Vercel 프로젝트 설정**

Vercel 대시보드에서:
1. Import Git Repository → `AIandMLcourse`
2. **Root Directory**: `week5-platform` (중요!)
3. Framework Preset: Next.js (자동 감지)
4. Environment Variables 추가:

```
NEXTAUTH_SECRET       = (week5/.env.local 값)
NEXTAUTH_URL          = https://week5-platform.vercel.app
GOOGLE_CLIENT_ID      = (Google Cloud Console 값)
GOOGLE_CLIENT_SECRET  = (Google Cloud Console 값)
POLAR_PRODUCT_ID      = (Polar.sh 값)
POLAR_ACCESS_TOKEN    = (Polar.sh 값)
POLAR_WEBHOOK_SECRET  = (Polar.sh 값)
TURSO_DATABASE_URL    = (Turso 값)
TURSO_AUTH_TOKEN      = (Turso 값)
NEXT_PUBLIC_FREE_USAGE_LIMIT = 5
```

5. Deploy 클릭

- [ ] **Step 4: Google OAuth Redirect URI 추가**

Google Cloud Console → OAuth 2.0 클라이언트 → 승인된 리디렉션 URI 추가:
```
https://week5-platform.vercel.app/api/auth/callback/google
```

- [ ] **Step 5: 프로덕션 빌드 로컬 확인**

```bash
npm run build
```
Expected: 빌드 에러 없음, 모든 5개 챕터 페이지 정적 생성 확인

- [ ] **Step 6: 최종 커밋 + 푸시**

```bash
git add week5-platform/vercel.json week5-platform/next.config.ts
git commit -m "feat: Vercel 배포 설정 완료"
git push origin main
```

Expected: Vercel이 자동으로 배포 시작 → `https://week5-platform.vercel.app` 에서 확인

---

## Self-Review

### Spec Coverage 확인

| 스펙 요구사항 | 구현 태스크 |
|---|---|
| Next.js 14 App Router | Task 1 |
| Tailwind CSS | Task 1, Task 5 |
| Google OAuth (NextAuth v5) | Task 4 |
| Demo 로그인 (mock session) | Task 4, Task 6 |
| Turso SQLite DB | Task 2 |
| 1~2챕터 무료, 3~5 유료 | Task 3, Task 8 |
| Polar.sh 샌드박스 결제 | Task 7 |
| Polar.sh 웹훅 → DB 반영 | Task 9 |
| Vercel 배포 (subfolder) | Task 10 |
| 챕터 상세 + 이전/다음 네비 | Task 8 |

### 주요 주의사항

1. **Turso DB 생성은 수동 필요** — Task 2 사전 작업 참고
2. **Google OAuth Redirect URI** — Vercel 배포 후 반드시 Google Cloud Console에 추가
3. **Polar.sh 웹훅 URL** — 배포 후 `https://<your-domain>/api/polar/webhook` 으로 업데이트
4. **NEXTAUTH_URL** — 로컬: `http://localhost:3000`, 배포: `https://week5-platform.vercel.app`
