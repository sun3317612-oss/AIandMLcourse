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
