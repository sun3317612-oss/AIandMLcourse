import { auth } from "@/auth"
import { chapters } from "@/lib/chapters"
import { upsertSubscription } from "@/lib/db"
import ChapterCard from "@/components/ChapterCard"
import PaymentButton from "@/components/PaymentButton"
import { Polar } from "@polar-sh/sdk"

const polar = new Polar({
  accessToken: process.env.POLAR_ACCESS_TOKEN!,
  server: "sandbox",
})

export default async function ChaptersPage({
  searchParams,
}: {
  searchParams: Promise<{ success?: string; checkout_id?: string }>
}) {
  const session = await auth()
  const params = await searchParams

  // 결제 성공 후 Polar API로 직접 확인해서 DB 업데이트
  if (params.success && params.checkout_id && session?.user?.id) {
    try {
      const checkout = await polar.checkouts.get(params.checkout_id)
      if (checkout.status === "succeeded") {
        await upsertSubscription({
          id: checkout.id,
          userId: session.user.id,
          polarSubscriptionId: checkout.id,
          status: "active",
        })
      }
    } catch {
      // 확인 실패 시 무시 (웹훅이 처리했을 수도 있음)
    }
  }

  // DB에서 최신 상태 다시 확인
  const { hasActiveSubscription, getUser } = await import("@/lib/db")
  let isPaid = false
  if (session?.user?.id) {
    try {
      const dbUser = await getUser(session.user.id)
      isPaid = dbUser
        ? Boolean(dbUser.is_demo) || (await hasActiveSubscription(session.user.id))
        : false
    } catch {
      isPaid = session?.user?.isPaid ?? false
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      <h1 className="text-3xl font-bold text-gray-900 mb-2">챕터 목록</h1>
      <p className="text-gray-500 mb-8">
        Week 5 딥러닝 핵심 개념 — 5개 챕터
      </p>

      {params.success && (
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
