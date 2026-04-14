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
