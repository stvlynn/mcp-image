import type { AuthRequest, OAuthHelpers } from '@cloudflare/workers-oauth-provider'
import { Hono } from 'hono'
import { accountCredentialsMatch, escapeHtml } from './credentials.js'

interface AuthEnv {
  OAUTH_PROVIDER: OAuthHelpers
  MCP_AUTH_USERNAME?: string
  MCP_AUTH_PASSWORD?: string
}

const app = new Hono<{ Bindings: AuthEnv }>()

function isAuthRequest(value: unknown): value is AuthRequest {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  if (!('responseType' in value) || typeof value.responseType !== 'string') {
    return false
  }
  if (!('clientId' in value) || typeof value.clientId !== 'string') {
    return false
  }
  if (!('redirectUri' in value) || typeof value.redirectUri !== 'string') {
    return false
  }
  if (!('state' in value) || typeof value.state !== 'string') {
    return false
  }
  if (!('scope' in value) || !Array.isArray(value.scope)) {
    return false
  }
  return value.scope.every((scope) => typeof scope === 'string')
}

function parseAuthState(state: string): AuthRequest | undefined {
  try {
    const parsed: unknown = JSON.parse(atob(state))
    return isAuthRequest(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

function cookieSuffix(request: Request): string {
  const secure = new URL(request.url).protocol === 'https:' ? '; Secure' : ''
  return `HttpOnly; Path=/; SameSite=Lax; Max-Age=600${secure}`
}

function readCookie(request: Request, name: string): string | undefined {
  const header = request.headers.get('Cookie') ?? ''
  const match = header
    .split(';')
    .map((part) => part.trim())
    .find((part) => part.startsWith(`${name}=`))
  return match?.slice(name.length + 1)
}

function loginPage(options: {
  clientName: string
  state: string
  csrfToken: string
  error?: string
}): string {
  const clientName = escapeHtml(options.clientName)
  const error = options.error ? `<p class="error">${escapeHtml(options.error)}</p>` : ''

  return `<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>登录</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 48px auto; max-width: 420px; padding: 0 20px; color: #1a1a1a; }
      form { display: grid; gap: 12px; border: 1px solid #e4e4e4; border-radius: 12px; padding: 24px; }
      label { display: grid; gap: 6px; font-size: 14px; }
      input { font: inherit; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px; }
      button { font: inherit; padding: 10px 12px; border: 0; border-radius: 8px; background: #111; color: #fff; }
      .error { color: #9b1c1c; margin: 0; }
      p { line-height: 1.5; }
    </style>
  </head>
  <body>
    <h1>账户登录</h1>
    <p><strong>${clientName}</strong> 请求访问图像生成 MCP。</p>
    <form method="POST" action="/authorize">
      ${error}
      <input type="hidden" name="state" value="${escapeHtml(options.state)}">
      <input type="hidden" name="csrf_token" value="${escapeHtml(options.csrfToken)}">
      <label>账户<input name="username" autocomplete="username" required></label>
      <label>密码<input name="password" type="password" autocomplete="current-password" required></label>
      <button type="submit">登录并授权</button>
    </form>
  </body>
</html>`
}

app.get('/authorize', async (c) => {
  const oauthReqInfo = await c.env.OAUTH_PROVIDER.parseAuthRequest(c.req.raw)
  const client = await c.env.OAUTH_PROVIDER.lookupClient(oauthReqInfo.clientId)
  if (!client) {
    return c.text('Invalid client_id', 400)
  }

  const csrfToken = crypto.randomUUID()
  c.header('Set-Cookie', `MCP_AUTH_CSRF=${csrfToken}; ${cookieSuffix(c.req.raw)}`)
  return c.html(
    loginPage({
      clientName: client.clientName || 'MCP client',
      state: btoa(JSON.stringify(oauthReqInfo)),
      csrfToken,
    })
  )
})

app.post('/authorize', async (c) => {
  const form = await c.req.formData()
  const state = form.get('state')
  const csrfToken = form.get('csrf_token')
  const username = form.get('username')
  const password = form.get('password')

  if (typeof state !== 'string' || typeof csrfToken !== 'string') {
    return c.text('Missing authorization state', 400)
  }
  if (csrfToken !== readCookie(c.req.raw, 'MCP_AUTH_CSRF')) {
    return c.text('Invalid CSRF token', 400)
  }

  const oauthReqInfo = parseAuthState(state)
  if (!oauthReqInfo) {
    return c.text('Invalid authorization state', 400)
  }

  const client = await c.env.OAUTH_PROVIDER.lookupClient(oauthReqInfo.clientId)
  if (!client) {
    return c.text('Invalid client_id', 400)
  }

  const usernameText = typeof username === 'string' ? username : ''
  const passwordText = typeof password === 'string' ? password : ''
  const authorized = await accountCredentialsMatch(
    c.env.MCP_AUTH_USERNAME,
    c.env.MCP_AUTH_PASSWORD,
    usernameText,
    passwordText
  )
  if (!authorized) {
    const csrf = crypto.randomUUID()
    c.header('Set-Cookie', `MCP_AUTH_CSRF=${csrf}; ${cookieSuffix(c.req.raw)}`)
    return c.html(
      loginPage({
        clientName: client.clientName || 'MCP client',
        state,
        csrfToken: csrf,
        error: '账户或密码不正确',
      }),
      401
    )
  }

  const { redirectTo } = await c.env.OAUTH_PROVIDER.completeAuthorization({
    request: oauthReqInfo,
    userId: usernameText,
    scope: oauthReqInfo.scope,
    metadata: {
      label: usernameText,
    },
    props: {
      username: usernameText,
    },
  })

  return c.redirect(redirectTo, 302)
})

app.get('/', (c) => {
  return c.text('mcp-image OAuth MCP. Connect a client to /mcp.')
})

export { app as AuthHandler }
