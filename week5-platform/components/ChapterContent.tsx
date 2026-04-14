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
