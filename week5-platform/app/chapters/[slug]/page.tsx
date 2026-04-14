import { notFound, redirect } from "next/navigation"
import Link from "next/link"
import { auth } from "@/auth"
import { getChapter, getPrevNext } from "@/lib/chapters"
import ChapterContent from "@/components/ChapterContent"

export default async function ChapterPage({
  params,
}: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const chapter = getChapter(slug)
  if (!chapter) notFound()

  // 유료 챕터 접근 제어
  if (!chapter.free) {
    const session = await auth()
    if (!session) redirect("/")
    if (!session.user.isPaid) redirect("/chapters")
  }

  const { prev, next } = getPrevNext(slug)

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
