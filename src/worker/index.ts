import { readFile } from 'node:fs/promises'
import { OAuthProvider } from '@cloudflare/workers-oauth-provider'
import { McpServer } from '@modelcontextprotocol/server'
import { createMcpHandler } from 'agents/mcp/server'
import { z } from 'zod'
import { MCPServerImpl } from '../server/mcpServer.js'
import type { McpToolResponse } from '../types/mcp.js'
import { AuthHandler } from './authHandler.js'

async function withSavedImage(result: McpToolResponse): Promise<McpToolResponse['content']> {
  if (result.isError) {
    return result.content
  }

  const embedded: McpToolResponse['content'] = []
  for (const item of result.content) {
    const image = await embedSavedFile(item.text)
    if (image) {
      embedded.push(image)
    }
  }
  return embedded.length === 0 ? result.content : [...result.content, ...embedded]
}

async function embedSavedFile(
  text: string
): Promise<McpToolResponse['content'][number] | undefined> {
  try {
    const resource = readSavedResource(text)
    if (!resource) {
      return undefined
    }
    const bytes = await readFile(filePathFromUri(resource.uri))
    return {
      type: 'text',
      text: JSON.stringify({
        type: 'image',
        mimeType: resource.mimeType,
        data: bytes.toString('base64'),
      }),
    }
  } catch {
    return undefined
  }
}

function readSavedResource(text: string): { uri: string; mimeType: string } | undefined {
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
    return undefined
  }
  return { uri: payload.resource.uri, mimeType: payload.resource.mimeType }
}

function filePathFromUri(uri: string): string {
  return uri.startsWith('file://') ? decodeURIComponent(new URL(uri).pathname) : uri
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
        background: z.enum(['transparent', 'opaque', 'auto']).optional(),
        inputFidelity: z.enum(['high', 'low']).optional(),
        moderation: z.enum(['low', 'auto']).optional(),
        outputFormat: z.enum(['png', 'jpeg', 'webp']).optional(),
        outputCompression: z.number().int().min(0).max(100).optional(),
        imageCount: z.number().int().min(1).max(10).optional(),
        maskImage: z.string().optional(),
        inputImages: z
          .array(z.object({ data: z.string(), mimeType: z.string().optional() }))
          .max(16)
          .optional(),
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
