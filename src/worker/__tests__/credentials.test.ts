import { describe, expect, it } from 'vitest'
import { accountCredentialsMatch } from '../credentials.js'

describe('accountCredentialsMatch', () => {
  it('accepts the configured account and password', async () => {
    await expect(accountCredentialsMatch('ada', 'secret', 'ada', 'secret')).resolves.toBe(true)
  })

  it('rejects a wrong password without accepting an empty configuration', async () => {
    await expect(accountCredentialsMatch('ada', 'secret', 'ada', 'other')).resolves.toBe(false)
    await expect(accountCredentialsMatch(undefined, undefined, '', '')).resolves.toBe(false)
    await expect(accountCredentialsMatch('ada', '', 'ada', '')).resolves.toBe(false)
  })
})
