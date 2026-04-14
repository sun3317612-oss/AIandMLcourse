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

  try {
    const checkout = await polar.checkouts.create({
      products: [process.env.POLAR_PRODUCT_ID!],
      successUrl: `${process.env.NEXTAUTH_URL}/chapters?success=true`,
      customerEmail: session.user?.email ?? undefined,
    })
    return NextResponse.json({ url: checkout.url })
  } catch (error) {
    console.error("Polar checkout error:", error)
    return NextResponse.json({ error: "Checkout failed" }, { status: 500 })
  }
}
