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
          ON CONFLICT(id) DO UPDATE SET name = :name, image = :image, is_demo = :is_demo`,
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
          VALUES (:id, :user_id, :polar_subscription_id, :status)
          ON CONFLICT(id) DO UPDATE SET status = :status`,
    args: {
      id: sub.id,
      user_id: sub.userId,
      polar_subscription_id: sub.polarSubscriptionId,
      status: sub.status,
    },
  })
}

export { db }
