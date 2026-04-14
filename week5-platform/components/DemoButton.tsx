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
