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
