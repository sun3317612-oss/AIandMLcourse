import NextAuth from "next-auth"
import Google from "next-auth/providers/google"
import Credentials from "next-auth/providers/credentials"
import { upsertUser, getUser, hasActiveSubscription } from "@/lib/db"

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    Credentials({
      id: "demo",
      name: "Demo",
      credentials: {},
      async authorize() {
        return {
          id: "demo-user-fixed",
          email: "demo@example.com",
          name: "Demo User",
          image: null,
          isDemo: true,
        }
      },
    }),
  ],
  callbacks: {
    async signIn({ user, account }) {
      const isDemo = account?.provider === "demo"
      await upsertUser({
        id: user.id!,
        email: user.email!,
        name: user.name,
        image: user.image,
        isDemo,
      })
      return true
    },
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id
        token.isDemo = (user as { isDemo?: boolean }).isDemo ?? false
      }
      return token
    },
    async session({ session, token }) {
      const dbUser = await getUser(token.id as string)
      const isPaid = dbUser
        ? Boolean(dbUser.is_demo) ||
          (await hasActiveSubscription(token.id as string))
        : false

      session.user.id = token.id as string
      session.user.isDemo = Boolean(token.isDemo)
      session.user.isPaid = isPaid
      return session
    },
  },
  pages: {
    signIn: "/",
  },
  secret: process.env.NEXTAUTH_SECRET,
})
