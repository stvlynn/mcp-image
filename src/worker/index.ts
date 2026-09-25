import { readFile } from 'node:fs/promises'
import { OAuthProvider } from '@cloudflare/workers-oauth-provider'
import { McpServer } from '@modelcontextprotocol/server'
import { createMcpHandler } from 'agents/mcp/server'
import { z } from 'zod'
import { MCPServerImpl } from '../server/mcpServer.js'
import type { McpToolResponse } from '../types/mcp.js'
import { AuthHandler } from './authHandler.js'

async function withSavedImage(result: McpToolResponse): Promise<McpToolResponse['content']> {
  const text = result.content[0]?.text
  if (!text || result.isError) {
    return result.content
  }

  try {
    const payload: unknown = JSON.parse(text)
    if (
      typeof payload !== 'object' ||
      payload === null ||
      !('resource' in payload) ||
      typeof payload.resource !== 'object' ||
      payload.resource === null ||
      !('uri' in payload.resource) ||
      typeof payload.resource.uri !== 'string' ||
      !('mimeType' in payload.resource) ||
      typeof payload.resource.mimeType !== 'string'
    ) {
      return result.content
    }

    const filePath = payload.resource.uri.startsWith('file://')
      ? decodeURIComponent(new URL(payload.resource.uri).pathname)
      : payload.resource.uri
    const bytes = await readFile(filePath)
    return [
      ...result.content,
      {
        type: 'text',
        text: JSON.stringify({
          type: 'image',
          mimeType: payload.resource.mimeType,
          data: bytes.toString('base64'),
        }),
      },
    ]
  } catch {
    return result.content
  }
}

function createImageServer(): McpServer {
  const impl = new MCPServerImpl()
  const tool = impl.getToolsList().tools[0]
  const server = new McpServer({
    name: 'mcp-image',
    version: impl.getServerInfo().version,
  })

  server.registerTool(
    'generate_image',
    {
      description: tool?.description ?? 'Generate or edit an image with OpenAI.',
      inputSchema: {
        prompt: z.string(),
        fileName: z.string().optional(),
        inputImage: z.string().optional(),
        inputImageMimeType: z.string().optional(),
        blendImages: z.boolean().optional(),
        maintainCharacterConsistency: z.boolean().optional(),
        useWorldKnowledge: z.boolean().optional(),
        aspectRatio: z.string().optional(),
        imageSize: z.string().optional(),
        purpose: z.string().optional(),
        quality: z.string().optional(),
      },
    },
    async (args) => {
      const result = await impl.callTool('generate_image', {
        ...args,
        provider: 'openai',
      })
      return {
        content: await withSavedImage(result),
        isError: result.isError,
      }
    }
  )

  return server
}

const mcpHandler = createMcpHandler(createImageServer)

const apiHandler = {
  fetch(request: Request, env: unknown, ctx: Parameters<typeof mcpHandler>[2]): Promise<Response> {
    return mcpHandler(request, env, ctx)
  },
}

export default new OAuthProvider({
  authorizeEndpoint: '/authorize',
  tokenEndpoint: '/token',
  clientRegistrationEndpoint: '/register',
  apiRoute: '/mcp',
  apiHandler,
  defaultHandler: AuthHandler,
})
