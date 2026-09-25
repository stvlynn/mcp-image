import * as path from 'node:path'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import {
  CallToolRequestSchema,
  type CallToolResult,
  ListToolsRequestSchema,
  type ListToolsResult,
} from '@modelcontextprotocol/sdk/types.js'
import packageJson from '../../package.json' with { type: 'json' }
import type { GeneratedImageResult, ImageApiParams, ImageClient } from '../api/imageClient.js'
import { generateFileName, readInputImage, saveImage } from '../business/fileManager.js'
import { validateBase64Image, validateGenerateImageParams } from '../business/inputValidator.js'
import { buildErrorResponse, buildSuccessResponse } from '../business/responseBuilder.js'
import {
  createStructuredPromptGenerator,
  type FeatureFlags,
  type StructuredPromptGenerator,
} from '../business/structuredPromptGenerator.js'
import type {
  GenerateImageParams,
  ImageOutputFormat,
  ImageProvider,
  MCPServerConfig,
  McpToolResponse,
} from '../types/mcp.js'
import {
  ASPECT_RATIO_VALUES,
  IMAGE_PROVIDER_VALUES,
  IMAGE_QUALITY_VALUES,
  IMAGE_SIZE_VALUES,
  OPENAI_BACKGROUND_VALUES,
  OPENAI_INPUT_FIDELITY_VALUES,
  OPENAI_MODERATION_VALUES,
  OPENAI_OUTPUT_FORMAT_VALUES,
} from '../types/mcp.js'
import { unwrapOrThrow } from '../types/result.js'
import { type Config, getConfig, validateProviderCredentials } from '../utils/config.js'
import { toError } from '../utils/errors.js'
import { Logger } from '../utils/logger.js'
import {
  reconcileFileNameExtension,
  resolvePreferredOutputFormat,
  SUPPORTED_EXTENSIONS,
} from '../utils/mimeUtils.js'
import { SecurityManager } from '../utils/security.js'
import { ErrorHandler } from './errorHandler.js'
import {
  getImageProviderDefinition,
  type ImageProviderDefinition,
} from './imageProviderRegistry.js'

const PACKAGE_VERSION = packageJson.version

const DEFAULT_CONFIG: MCPServerConfig = {
  name: 'mcp-image-server',
  version: PACKAGE_VERSION,
  defaultOutputDir: './output',
}

interface ProviderClients {
  imageClient: ImageClient
  structuredPromptGenerator: StructuredPromptGenerator | null
}

type ImageOptions = Omit<ImageApiParams, 'prompt'>

function rejectUnsupportedOpenAIOptions(
  providerName: ImageProvider,
  params: GenerateImageParams
): void {
  if (providerName !== 'openai') {
    assertNoOpenAIOnlyOptions(params)
  }
}

async function saveGeneratedImages(
  generatedImage: GeneratedImageResult,
  fileName: string,
  outputDir: string,
  securityManager: SecurityManager
): Promise<McpToolResponse> {
  const images = [generatedImage.imageData, ...(generatedImage.extraImageData ?? [])]
  const content: McpToolResponse['content'] = []
  for (const [index, imageData] of images.entries()) {
    const indexedName = images.length === 1 ? fileName : indexedFileName(fileName, index)
    const outputPath = path.join(outputDir, indexedName)
    const sanitizedPath = unwrapOrThrow(securityManager.sanitizeFilePath(outputPath))
    const savedPath = unwrapOrThrow(await saveImage(imageData, sanitizedPath))
    content.push(...buildSuccessResponse(generatedImage, savedPath).content)
  }
  return { content, isError: false }
}

function assertNoOpenAIOnlyOptions(params: GenerateImageParams): void {
  const used = [
    params.background,
    params.inputFidelity,
    params.moderation,
    params.outputCompression,
    params.imageCount,
    params.outputFormat,
    params.maskImage,
    params.maskImagePath,
    params.inputImages,
  ].some((value) => value !== undefined)
  if (used) {
    throw new Error(
      'background, inputFidelity, moderation, outputCompression, imageCount, outputFormat, maskImage, and inputImages are supported only when provider is openai'
    )
  }
}

async function loadMaskImage(params: GenerateImageParams): Promise<string | undefined> {
  if (params.maskImagePath) {
    const mask = await readInputImage(params.maskImagePath)
    return mask.data.toString('base64')
  }
  if (!params.maskImage) {
    return undefined
  }
  const decoded = unwrapOrThrow(validateBase64Image(params.maskImage, 'image/png'))
  return decoded?.toString('base64')
}

function indexedFileName(fileName: string, index: number): string {
  const extension = path.extname(fileName)
  const stem = extension ? fileName.slice(0, -extension.length) : fileName
  return `${stem}-${index + 1}${extension}`
}

/** Load an edit source from a local path or an inline base64 payload. */
async function loadInputImage(
  params: GenerateImageParams
): Promise<{ data?: string; mimeType?: string }> {
  if (params.inputImagePath) {
    const inputImage = await readInputImage(params.inputImagePath)
    return {
      data: inputImage.data.toString('base64'),
      mimeType: inputImage.mimeType,
    }
  }

  if (!params.inputImage) {
    return {}
  }

  const decoded = unwrapOrThrow(validateBase64Image(params.inputImage, params.inputImageMimeType))
  if (!decoded) {
    return {}
  }

  return {
    data: decoded.toString('base64'),
    ...(params.inputImageMimeType ? { mimeType: params.inputImageMimeType } : {}),
  }
}

/** Assemble the provider-facing image options from validated request params. */
function buildImageOptions(
  params: GenerateImageParams,
  inputImage: { data?: string; mimeType?: string },
  preferredOutputFormat: ImageOutputFormat | undefined
): ImageOptions {
  return {
    ...(inputImage.data && { inputImage: inputImage.data }),
    ...(inputImage.mimeType && { inputImageMimeType: inputImage.mimeType }),
    ...(params.aspectRatio && { aspectRatio: params.aspectRatio }),
    ...(params.imageSize && { imageSize: params.imageSize }),
    ...(params.useGoogleSearch !== undefined && { useGoogleSearch: params.useGoogleSearch }),
    ...(preferredOutputFormat && { preferredOutputFormat }),
    ...(params.quality !== undefined && { quality: params.quality }),
    ...(params.background && { background: params.background }),
    ...(params.inputFidelity && { inputFidelity: params.inputFidelity }),
    ...(params.moderation && { moderation: params.moderation }),
    ...(params.outputCompression !== undefined && { outputCompression: params.outputCompression }),
    ...(params.imageCount !== undefined && { imageCount: params.imageCount }),
    ...(params.outputFormat && { outputFormat: params.outputFormat }),
    ...(params.inputImages && { inputImages: params.inputImages }),
  } satisfies ImageOptions
}

/** Project the request's enhancement flags onto the prompt generator's contract. */
function buildFeatureFlags(params: GenerateImageParams): FeatureFlags {
  return {
    ...(params.maintainCharacterConsistency !== undefined && {
      maintainCharacterConsistency: params.maintainCharacterConsistency,
    }),
    ...(params.blendImages !== undefined && { blendImages: params.blendImages }),
    ...(params.useWorldKnowledge !== undefined && { useWorldKnowledge: params.useWorldKnowledge }),
  }
}

/**
 * Decide the saved file name. A caller-supplied name keeps its stem but takes
 * the extension of the image that was actually generated; `corrected` reports
 * whether a supported requested extension was replaced.
 */
function resolveOutputFileName(
  requestedFileName: string | undefined,
  sanitizedFileName: string | undefined,
  mimeType: string
): { fileName: string; requestedExtension: string; corrected: boolean } {
  const rawFileName = sanitizedFileName ?? generateFileName(mimeType)
  const fileName = requestedFileName
    ? reconcileFileNameExtension(rawFileName, mimeType)
    : rawFileName
  const requestedExtension = path.extname(rawFileName)
  return {
    fileName,
    requestedExtension,
    corrected:
      sanitizedFileName !== undefined &&
      fileName !== rawFileName &&
      SUPPORTED_EXTENSIONS.includes(requestedExtension.toLowerCase()),
  }
}

export class MCPServerImpl {
  private config: MCPServerConfig
  private server: Server | null = null
  private logger: Logger
  private securityManager: SecurityManager
  private clientsByProvider = new Map<ImageProvider, ProviderClients>()

  constructor(config: Partial<MCPServerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    this.logger = new Logger()
    this.securityManager = new SecurityManager()
  }

  public getServerInfo(): { name: string; version: string } {
    return {
      name: this.config.name,
      version: this.config.version,
    }
  }

  public getToolsList(): ListToolsResult {
    return {
      tools: [
        {
          name: 'generate_image',
          description:
            'Generate a new image from a text prompt or edit an existing image using inputImagePath. Saves the result and returns a file resource.',
          inputSchema: {
            type: 'object' as const,
            properties: {
              prompt: {
                type: 'string' as const,
                description:
                  'Describe the image to generate or the edit to apply. Include the subject, context, and visual style; English is recommended for prompt enhancement.',
              },
              fileName: {
                type: 'string' as const,
                description:
                  'Use .png, .jpg, or .jpeg to request that output format from OpenAI or Seedream. Other or absent suffixes use the provider default; the saved filename is corrected to the actual image extension.',
              },
              inputImagePath: {
                type: 'string' as const,
                description:
                  'Provide an absolute path to a source image when editing, creating a variation, or transferring style.',
              },
              blendImages: {
                type: 'boolean' as const,
                description:
                  'Enable when the prompt combines multiple visual elements that need coherent spatial relationships, lighting, or composition.',
              },
              maintainCharacterConsistency: {
                type: 'boolean' as const,
                description:
                  'Enable when the same character must retain a recognizable appearance across poses or scenes.',
              },
              useWorldKnowledge: {
                type: 'boolean' as const,
                description:
                  'Enable when accurate real-world details matter, such as historical figures, landmarks, cultures, or factual settings.',
              },
              useGoogleSearch: {
                type: 'boolean' as const,
                description:
                  'Enable when using Gemini and the image requires current or time-sensitive web information. With OpenAI or Seedream, omit this option or set it to false.',
              },
              aspectRatio: {
                type: 'string' as const,
                description:
                  'Set the requested output aspect ratio. Omit to use the provider default. OpenAI does not support 1:4, 1:8, 4:1, or 8:1.',
                enum: [...ASPECT_RATIO_VALUES],
              },
              imageSize: {
                type: 'string' as const,
                description:
                  "Set the requested output size to 1K, 2K, or 4K. Omit to use the selected provider and quality preset's default. With Seedream, use 1K or 2K.",
                enum: [...IMAGE_SIZE_VALUES],
              },
              purpose: {
                type: 'string' as const,
                description:
                  "Describe the image's intended use, such as a cookbook cover, social media post, or presentation slide, so prompt enhancement can adapt composition and detail.",
              },
              quality: {
                type: 'string' as const,
                description:
                  'Set only when the user requests a quality level; otherwise omit to use the server default. fast prioritizes speed, balanced trades speed for detail, and quality prioritizes fidelity.',
                enum: [...IMAGE_QUALITY_VALUES],
              },
              provider: {
                type: 'string' as const,
                description:
                  'Set only when the user requests a specific image provider; otherwise omit to use the server default. The provider must have its API key configured on the server.',
                enum: [...IMAGE_PROVIDER_VALUES],
              },
              background: {
                type: 'string' as const,
                description:
                  'OpenAI only. transparent, opaque, or auto. Transparent output requires PNG or WebP.',
                enum: [...OPENAI_BACKGROUND_VALUES],
              },
              inputFidelity: {
                type: 'string' as const,
                description:
                  'OpenAI edits only. high keeps facial features and style closer to the input images. low is the default.',
                enum: [...OPENAI_INPUT_FIDELITY_VALUES],
              },
              moderation: {
                type: 'string' as const,
                description: 'OpenAI generation only. low or auto.',
                enum: [...OPENAI_MODERATION_VALUES],
              },
              outputFormat: {
                type: 'string' as const,
                description: 'OpenAI only. png, jpeg, or webp. Overrides the filename extension.',
                enum: [...OPENAI_OUTPUT_FORMAT_VALUES],
              },
              outputCompression: {
                type: 'integer' as const,
                description: 'OpenAI JPEG or WebP compression from 0 to 100.',
                minimum: 0,
                maximum: 100,
              },
              imageCount: {
                type: 'integer' as const,
                description: 'OpenAI only. Number of images to return, from 1 to 10.',
                minimum: 1,
                maximum: 10,
              },
              maskImage: {
                type: 'string' as const,
                description:
                  'OpenAI edits only. Base64 PNG mask. Transparent pixels mark the area to edit.',
              },
              maskImagePath: {
                type: 'string' as const,
                description: 'OpenAI edits only. Absolute path to a PNG mask.',
              },
              inputImages: {
                type: 'array' as const,
                description:
                  'OpenAI edits only. Up to 16 reference images, including inputImage when both are set. Each item is { data, mimeType }.',
                items: {
                  type: 'object' as const,
                  properties: {
                    data: { type: 'string' as const },
                    mimeType: { type: 'string' as const },
                  },
                  required: ['data'],
                },
              },
            },
            required: ['prompt'],
          },
        },
      ],
    }
  }

  public async callTool(
    name: string,
    args: unknown,
    signal?: AbortSignal
  ): Promise<McpToolResponse> {
    try {
      if (name === 'generate_image') {
        return await this.handleGenerateImage(args, signal)
      }
      throw new Error(`Unknown tool: ${name}`)
    } catch (error) {
      const toolError = toError(error)
      this.logger.error('mcp-server', 'Tool execution failed', toolError)
      return ErrorHandler.handleError(toolError)
    }
  }

  /**
   * Initialize provider clients lazily, cached per provider so that requests
   * alternating between providers do not reuse another provider's clients.
   */
  private getProviderClients(
    config: Config,
    providerName: ImageProvider,
    provider: ImageProviderDefinition
  ): ProviderClients {
    const cached = this.clientsByProvider.get(providerName)
    if (cached && (config.skipPromptEnhancement || cached.structuredPromptGenerator)) {
      return cached
    }

    const structuredPromptGenerator = config.skipPromptEnhancement
      ? null
      : createStructuredPromptGenerator(
          provider.createTextClient(config),
          provider.promptGeneration.maxTokens
        )

    const clients: ProviderClients = {
      imageClient: cached?.imageClient ?? provider.createImageClient(config),
      structuredPromptGenerator,
    }
    this.clientsByProvider.set(providerName, clients)

    this.logger.info('mcp-server', 'Image provider clients initialized', {
      provider: providerName,
      promptEnhancement: !config.skipPromptEnhancement,
    })

    return clients
  }

  /**
   * Send the prompt for enhancement and report the outcome. A failed
   * enhancement is not fatal: the original prompt is used instead.
   */
  private async enhancePrompt(
    generator: StructuredPromptGenerator,
    params: GenerateImageParams,
    context: { inputImageData?: string; inputImageMimeType?: string; signal?: AbortSignal }
  ): Promise<string> {
    const promptResult = await generator.generateStructuredPrompt(params.prompt, {
      features: buildFeatureFlags(params),
      ...(context.inputImageData !== undefined && { inputImageData: context.inputImageData }),
      ...(params.purpose !== undefined && { purpose: params.purpose }),
      ...(context.inputImageMimeType !== undefined && {
        inputImageMimeType: context.inputImageMimeType,
      }),
      ...(context.signal !== undefined && { signal: context.signal }),
    })
    context.signal?.throwIfAborted()

    if (!promptResult.success) {
      this.logger.warn('mcp-server', 'Using original prompt', {
        error: promptResult.error.message,
      })
      return params.prompt
    }

    this.logger.info('mcp-server', 'Structured prompt generated', {
      originalLength: params.prompt.length,
      structuredLength: promptResult.data.length,
    })
    return promptResult.data
  }

  private async handleGenerateImage(args: unknown, signal?: AbortSignal): Promise<McpToolResponse> {
    const result = await ErrorHandler.wrapWithResultType(
      () => this.generateImage(args, signal),
      'image-generation'
    )

    if (result.success) {
      return result.data
    }

    return buildErrorResponse(result.error)
  }

  /**
   * One image generation, in order: validate, resolve config and provider,
   * load any input image, enhance the prompt, generate, then save. Runs inside
   * the caller's error boundary, so a failed step throws.
   */
  private async generateImage(args: unknown, signal?: AbortSignal): Promise<McpToolResponse> {
    signal?.throwIfAborted()
    const params = unwrapOrThrow(validateGenerateImageParams(args))

    const sanitizedFileName = params.fileName
      ? this.securityManager.sanitizeFilename(params.fileName)
      : undefined
    const preferredOutputFormat = resolvePreferredOutputFormat(sanitizedFileName)

    const config = unwrapOrThrow(getConfig())

    // Reject an unusable output path before anything is sent upstream.
    unwrapOrThrow(
      this.securityManager.sanitizeFilePath(
        path.join(config.imageOutputDir, sanitizedFileName ?? 'image.png')
      )
    )

    const providerName = params.provider ?? config.imageProvider
    unwrapOrThrow(validateProviderCredentials(config, providerName))
    const provider = getImageProviderDefinition(providerName)

    const { imageClient, structuredPromptGenerator } = this.getProviderClients(
      config,
      providerName,
      provider
    )

    rejectUnsupportedOpenAIOptions(providerName, params)

    const inputImage = await loadInputImage(params)
    const inputImageData = inputImage.data
    const inputImageMimeType = inputImage.mimeType
    const maskImage = await loadMaskImage(params)

    const imageOptions = {
      ...buildImageOptions(
        params,
        {
          ...(inputImageData && { data: inputImageData }),
          ...(inputImageMimeType && { mimeType: inputImageMimeType }),
        },
        preferredOutputFormat
      ),
      ...(maskImage && { maskImage }),
    }

    provider.validateImageOptions?.(imageOptions, config)
    signal?.throwIfAborted()

    let structuredPrompt = params.prompt
    if (!config.skipPromptEnhancement && structuredPromptGenerator) {
      structuredPrompt = await this.enhancePrompt(structuredPromptGenerator, params, {
        ...(inputImageData !== undefined && { inputImageData }),
        ...(inputImageMimeType !== undefined && { inputImageMimeType }),
        ...(signal !== undefined && { signal }),
      })
    } else if (config.skipPromptEnhancement) {
      this.logger.info('mcp-server', 'Prompt enhancement skipped (SKIP_PROMPT_ENHANCEMENT=true)')
    }

    const generationResult = await imageClient.generateImage({
      prompt: structuredPrompt,
      ...imageOptions,
      ...(signal && { signal }),
    })
    signal?.throwIfAborted()

    const generatedImage = unwrapOrThrow(generationResult)
    const mimeType = generatedImage.metadata.mimeType
    const { fileName, requestedExtension, corrected } = resolveOutputFileName(
      params.fileName,
      sanitizedFileName,
      mimeType
    )
    if (corrected) {
      this.logger.warn(
        'mcp-server',
        'Output filename extension corrected to match generated MIME type',
        {
          requestedExtension,
          savedExtension: path.extname(fileName),
          mimeType,
        }
      )
    }
    return saveGeneratedImages(
      generatedImage,
      fileName,
      config.imageOutputDir,
      this.securityManager
    )
  }

  public initialize(): Server {
    this.server = new Server(
      {
        name: this.config.name,
        version: this.config.version,
      },
      {
        capabilities: {
          tools: {},
        },
      }
    )

    this.setupHandlers()

    return this.server
  }

  private setupHandlers(): void {
    if (!this.server) {
      throw new Error('Server not initialized')
    }

    this.server.setRequestHandler(ListToolsRequestSchema, async (): Promise<ListToolsResult> => {
      return this.getToolsList()
    })

    this.server.setRequestHandler(
      CallToolRequestSchema,
      async (request, { signal }): Promise<CallToolResult> => {
        const { name, arguments: args } = request.params
        const result = await this.callTool(name, args, signal)
        const response: CallToolResult = {
          content: result.content,
          isError: result.isError,
        }
        return response
      }
    )
  }
}

export function createMCPServer(config: Partial<MCPServerConfig> = {}): MCPServerImpl {
  return new MCPServerImpl(config)
}
