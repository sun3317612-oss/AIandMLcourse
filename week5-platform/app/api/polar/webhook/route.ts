import { NextRequest, NextResponse } from "next/server"
import { validateEvent, WebhookVerificationError } from "@polar-sh/sdk/webhooks"
import { upsertSubscription } from "@/lib/db"

export async function POST(req: NextRequest) {
  const body = await req.text()

  // Convert Next.js Headers to plain Record<string, string> required by validateEvent
  const headers: Record<string, string> = {}
  req.headers.forEach((value, key) => {
    headers[key] = value
  })

  let event: Awaited<ReturnType<typeof validateEvent>>
  try {
    event = validateEvent(body, headers, process.env.POLAR_WEBHOOK_SECRET!)
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
    const userId =
      typeof sub.metadata?.user_id === "string"
        ? sub.metadata.user_id
        : undefined

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
