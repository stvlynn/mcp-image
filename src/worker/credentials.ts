import { timingSafeEqual } from 'node:crypto'

async function digest(value: string): Promise<Uint8Array> {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))
  return new Uint8Array(hash)
}

async function digestsMatch(left: string, right: string): Promise<boolean> {
  const [leftDigest, rightDigest] = await Promise.all([digest(left), digest(right)])
  return timingSafeEqual(leftDigest, rightDigest)
}

/**
 * Compare the submitted account with the Worker env credentials.
 * Both fields are hashed before comparison so a mismatch does not return early.
 * Empty configured credentials never match.
 */
export async function accountCredentialsMatch(
  expectedUsername: string | undefined,
  expectedPassword: string | undefined,
  username: string,
  password: string
): Promise<boolean> {
  if (!expectedUsername || !expectedPassword) {
    return false
  }

  const [usernameMatches, passwordMatches] = await Promise.all([
    digestsMatch(expectedUsername, username),
    digestsMatch(expectedPassword, password),
  ])
  return usernameMatches && passwordMatches
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}
