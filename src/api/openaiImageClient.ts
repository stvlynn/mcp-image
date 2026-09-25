import OpenAI, { toFile } from 'openai'
import type {
  ImageEditParamsNonStreaming,
  ImageGenerateParamsNonStreaming,
  ImagesResponse,
} from 'openai/resources/images'
import type { ImageQuality } from '../types/mcp.js'
import { ASPECT_RATIO_VALUES } from '../types/mcp.js'
import type { Result } from '../types/result.js'
import { Err, Ok } from '../types/result.js'
import type { Config } from '../utils/config.js'
import { ImageAPIError, NetworkError } from '../utils/errors.js'
import {
  DEFAULT_MIME_TYPE,
  matchesImageDataMimeType,
  normalizeMimeType,
} from '../utils/mimeUtils.js'
import { extractStatusCode, isNetworkError } from './errorClassification.js'
import type { GeneratedImageResult, ImageApiParams, ImageClient } from './imageClient.js'

type OpenAIImageSize = `${number}x${number}`
type OpenAIImageQuality = 'low' | 'high' | 'max'
// GPT Image 2.5 supports flexible resolutions and max quality, while SDK types
// still enumerate older sizes and quality settings. Keep this compatibility
// boundary local: OpenAIImagesApi accepts SDK requests plus the documented 2.5 variant.
type OpenAIImageGenerateRequest = Omit<ImageGenerateParamsNonStreaming, 'size' | 'quality'> & {
  size: OpenAIImageSize
  quality: OpenAIImageQuality
}
type OpenAIImageEditRequest = Omit<ImageEditParamsNonStreaming, 'size' | 'quality'> & {
  size: OpenAIImageSize
  quality: OpenAIImageQuality
}

interface OpenAIImagesApi {
  generate(
    body: ImageGenerateParamsNonStreaming | OpenAIImageGenerateRequest,
    options?: { signal?: AbortSignal }
  ): Promise<ImagesResponse>
  edit(
    body: ImageEditParamsNonStreaming | OpenAIImageEditRequest,
    options?: { signal?: AbortSignal }
  ): Promise<ImagesResponse>
}
type ImageEditApiParams = ImageApiParams & { inputImage: string }

function mapQuality(quality: ImageQuality): OpenAIImageQuality {
  switch (quality) {
    case 'quality':
      return 'max'
    case 'balanced':
      return 'high'
    case 'fast':
      return 'low'
  }
}

function requestedLongEdge(imageSize: ImageApiParams['imageSize'], ratio: number): number {
  if (imageSize === '4K') {
    return 3840
  }
  if (imageSize === '2K') {
    return 2048
  }
  return ratio === 1 ? 1024 : 1536
}

function mapSize(params: ImageApiParams): OpenAIImageSize {
  const [width = 1, height = 1] = (params.aspectRatio ?? '1:1').split(':').map(Number)
  const ratio = Math.max(width, height) / Math.min(width, height)
  const requestedEdge = requestedLongEdge(params.imageSize, ratio)
  // GPT Image 2.5: 16px increments, at most 3840px per edge and 8,294,400 pixels.
  // Flooring keeps the pixel cap intact even for near-square 4K requests.
  const longEdge = Math.floor(Math.min(requestedEdge, Math.sqrt(8_294_400 * ratio)) / 16) * 16
  const shortEdge = Math.floor(longEdge / ratio / 16) * 16
  return width >= height ? `${longEdge}x${shortEdge}` : `${shortEdge}x${longEdge}`
}

function mimeTypeToExtension(mimeType: string): string {
  switch (normalizeMimeType(mimeType)) {
    case 'image/jpeg':
      return 'jpg'
    case 'image/webp':
      return 'webp'
    default:
      return 'png'
  }
}

const OPENAI_IMAGE_MODELS = {
  FLARE: 'gpt-image-2.5-flare',
  SUNBURST: 'gpt-image-2.5-sunburst',
} as const

async function readGeneratedImageBytes(
  image: { b64_json?: string | null; url?: string | null } | undefined,
  signal?: AbortSignal
): Promise<Buffer | undefined> {
  if (image?.b64_json) {
    return Buffer.from(image.b64_json, 'base64')
  }
  if (!image?.url) {
    return undefined
  }

  const response = await fetch(image.url, {
    ...(signal ? { signal } : {}),
  })
  if (!response.ok) {
    throw new Error(`Image URL download failed with status ${response.status}`)
  }
  return Buffer.from(await response.arrayBuffer())
}

function hasInputImage(params: ImageApiParams): params is ImageEditApiParams {
  return typeof params.inputImage === 'string' && params.inputImage.length > 0
}

function validateOpenAIRequestOptions(
  params: ImageApiParams,
  outputFormat: 'png' | 'jpeg' | 'webp'
): ImageAPIError | undefined {
  const editing = hasInputImage(params) || (params.inputImages?.length ?? 0) > 0
  if ((params.maskImage || params.inputFidelity) && !editing) {
    return new ImageAPIError(
      'OpenAI mask and inputFidelity require an input image',
      'Provide inputImage, inputImagePath, or inputImages when editing'
    )
  }
  if (params.moderation && editing) {
    return new ImageAPIError(
      'OpenAI moderation applies to image generation, not editing',
      'Omit moderation when sending an input image'
    )
  }
  if (params.outputCompression !== undefined && outputFormat === 'png') {
    return new ImageAPIError(
      'OpenAI outputCompression applies to JPEG and WebP output',
      'Set outputFormat to jpeg or webp before using outputCompression'
    )
  }
  if (params.background === 'transparent' && outputFormat === 'jpeg') {
    return new ImageAPIError(
      'Transparent OpenAI backgrounds require PNG or WebP output',
      'Set outputFormat to png or webp when background is transparent'
    )
  }
  const imageCount = (hasInputImage(params) ? 1 : 0) + (params.inputImages?.length ?? 0)
  if (imageCount > 16) {
    return new ImageAPIError(
      'OpenAI image edits accept at most 16 input images',
      'Send no more than 16 images in inputImage and inputImages combined'
    )
  }
  return undefined
}

async function uploadImage(
  data: string,
  mimeType: string | undefined,
  stem: string
): Promise<Awaited<ReturnType<typeof toFile>>> {
  const normalized = normalizeMimeType(mimeType ?? DEFAULT_MIME_TYPE)
  return toFile(Buffer.from(data, 'base64'), `${stem}.${mimeTypeToExtension(normalized)}`, {
    type: normalized,
  })
}

async function assembleGeneratedImages(
  images: ImagesResponse['data'],
  outputFormat: 'png' | 'jpeg' | 'webp',
  modelName: string,
  signal?: AbortSignal
): Promise<Result<{ imageData: Buffer; extraImageData: Buffer[] }, ImageAPIError | NetworkError>> {
  const imageBytes = await readAllGeneratedImages(images, signal)
  if (!imageBytes || imageBytes.some((bytes) => !matchesOutputFormat(bytes, outputFormat))) {
    return Err(
      new ImageAPIError(
        imageBytes
          ? 'OpenAI image response did not match the requested output format'
          : 'No image data returned from OpenAI image API',
        {
          provider: 'openai',
          model: modelName,
          stage: imageBytes ? 'image_response' : 'image_extraction',
          suggestion: imageBytes
            ? 'Retry the request; the provider returned unexpected image bytes'
            : 'Retry the request or verify that the selected model returns base64 image data or an image URL',
        }
      )
    )
  }
  const [imageData, ...extraImageData] = imageBytes
  if (!imageData) {
    return Err(
      new ImageAPIError('No image data returned from OpenAI image API', {
        provider: 'openai',
        model: modelName,
        stage: 'image_extraction',
        suggestion:
          'Retry the request or verify that the selected model returns base64 image data or an image URL',
      })
    )
  }
  return Ok({ imageData, extraImageData })
}

async function readAllGeneratedImages(
  images: ImagesResponse['data'],
  signal?: AbortSignal
): Promise<Buffer[] | undefined> {
  if (!images || images.length === 0) {
    return undefined
  }
  const buffers: Buffer[] = []
  for (const image of images) {
    const bytes = await readGeneratedImageBytes(image, signal)
    if (!bytes) {
      return undefined
    }
    buffers.push(bytes)
  }
  return buffers
}

function mimeTypeForOutputFormat(format: 'png' | 'jpeg' | 'webp'): string {
  if (format === 'jpeg') {
    return 'image/jpeg'
  }
  if (format === 'webp') {
    return 'image/webp'
  }
  return 'image/png'
}

function matchesOutputFormat(imageData: Buffer, format: 'png' | 'jpeg' | 'webp'): boolean {
  if (format === 'webp') {
    return (
      imageData.length >= 12 &&
      imageData.subarray(0, 4).toString('ascii') === 'RIFF' &&
      imageData.subarray(8, 12).toString('ascii') === 'WEBP'
    )
  }
  return matchesImageDataMimeType(imageData, format === 'jpeg' ? 'image/jpeg' : 'image/png')
}

export function validateOpenAIOptions(
  params: Pick<ImageApiParams, 'useGoogleSearch' | 'aspectRatio'>
): Result<true, ImageAPIError> {
  if (params.useGoogleSearch) {
    return Err(
      new ImageAPIError(
        'useGoogleSearch is not supported by the OpenAI image provider',
        'Disable useGoogleSearch or use IMAGE_PROVIDER=gemini for Google Search grounding'
      )
    )
  }

  if (params.aspectRatio !== undefined) {
    const [width = 0, height = 0] = params.aspectRatio.split(':').map(Number)
    if (
      !ASPECT_RATIO_VALUES.includes(params.aspectRatio) ||
      Math.max(width, height) / Math.min(width, height) > 3
    ) {
      return Err(
        new ImageAPIError(
          'Unsupported OpenAI image aspect ratio',
          'Use a supported aspect ratio between 1:3 and 3:1'
        )
      )
    }
  }

  return Ok(true)
}

class OpenAIImageClientImpl implements ImageClient {
  constructor(
    private readonly client: OpenAI,
    private readonly defaultQuality: ImageQuality = 'fast'
  ) {}

  async generateImage(
    params: ImageApiParams
  ): Promise<Result<GeneratedImageResult, ImageAPIError | NetworkError>> {
    try {
      const optionsResult = validateOpenAIOptions(params)
      if (!optionsResult.success) {
        return optionsResult
      }

      const effectiveQuality = params.quality ?? this.defaultQuality
      const modelName =
        effectiveQuality === 'quality' ? OPENAI_IMAGE_MODELS.SUNBURST : OPENAI_IMAGE_MODELS.FLARE
      const quality = mapQuality(effectiveQuality)
      const size = mapSize(params)
      const outputFormat = params.outputFormat ?? params.preferredOutputFormat ?? 'png'
      const response = await this.requestImage(params, {
        model: modelName,
        prompt: params.prompt,
        n: params.imageCount ?? 1,
        output_format: outputFormat,
        quality,
        size,
      })
      if ('success' in response) {
        return response
      }

      const assembled = await assembleGeneratedImages(
        response.data,
        outputFormat,
        modelName,
        params.signal
      )
      if (!assembled.success) {
        return assembled
      }
      const { imageData, extraImageData } = assembled.data
      const mimeType = mimeTypeForOutputFormat(outputFormat)

      const firstImage = response.data?.[0]
      return Ok({
        imageData,
        ...(extraImageData.length > 0 && { extraImageData }),
        metadata: {
          model: modelName,
          provider: 'openai',
          prompt: params.prompt,
          mimeType,
          timestamp: new Date(),
          inputImageProvided: hasInputImage(params) || (params.inputImages?.length ?? 0) > 0,
          ...(firstImage?.revised_prompt && { revisedPrompt: firstImage.revised_prompt }),
        },
      })
    } catch (error) {
      return this.handleError(error, params.prompt)
    }
  }

  private async requestImage(
    params: ImageApiParams,
    request: OpenAIImageGenerateRequest
  ): Promise<ImagesResponse | Result<never, ImageAPIError>> {
    const outputFormat = request.output_format ?? 'png'
    const optionError = validateOpenAIRequestOptions(params, outputFormat)
    if (optionError) {
      return Err(optionError)
    }
    const body: OpenAIImageGenerateRequest = {
      ...request,
      ...(params.background && { background: params.background }),
      ...(params.outputCompression !== undefined && {
        output_compression: params.outputCompression,
      }),
    }
    const editing = hasInputImage(params) || (params.inputImages?.length ?? 0) > 0
    if (editing) {
      return this.editImage(params, body)
    }
    const images: OpenAIImagesApi = this.client.images
    return images.generate(
      {
        ...body,
        ...(params.moderation && { moderation: params.moderation }),
      },
      {
        ...(params.signal && { signal: params.signal }),
      }
    )
  }

  private async editImage(
    params: ImageApiParams,
    request: OpenAIImageGenerateRequest
  ): Promise<ImagesResponse> {
    const sources = [
      ...(hasInputImage(params)
        ? [{ data: params.inputImage, mimeType: params.inputImageMimeType }]
        : []),
      ...(params.inputImages ?? []),
    ]
    const files = await Promise.all(
      sources.map((source, index) => uploadImage(source.data, source.mimeType, `input-${index}`))
    )
    const firstFile = files[0]
    if (!firstFile) {
      throw new Error('OpenAI image edit requires an input image')
    }
    const image = files.length === 1 ? firstFile : files
    const mask = params.maskImage
      ? await uploadImage(params.maskImage, 'image/png', 'mask')
      : undefined

    const images: OpenAIImagesApi = this.client.images
    return await images.edit(
      {
        ...request,
        image,
        ...(mask && { mask }),
        ...(params.inputFidelity && { input_fidelity: params.inputFidelity }),
      },
      {
        ...(params.signal && { signal: params.signal }),
      }
    )
  }

  private handleError(error: unknown, prompt: string): Result<never, ImageAPIError | NetworkError> {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error'

    if (isNetworkError(error)) {
      return Err(
        new NetworkError(
          'Network error during OpenAI image generation',
          'Check your internet connection and try again',
          error instanceof Error ? error : undefined
        )
      )
    }

    return Err(
      new ImageAPIError(
        'Failed to generate image with OpenAI',
        {
          provider: 'openai',
          prompt,
          upstreamMessage: errorMessage,
          suggestion: this.getAPIErrorSuggestion(errorMessage),
        },
        extractStatusCode(error)
      )
    )
  }

  private getAPIErrorSuggestion(errorMessage: string): string {
    const lowerMessage = errorMessage.toLowerCase()

    if (lowerMessage.includes('quota') || lowerMessage.includes('rate limit')) {
      return 'You have exceeded your OpenAI API quota or rate limit. Wait before retrying or upgrade your plan'
    }

    if (lowerMessage.includes('unauthorized') || lowerMessage.includes('api key')) {
      return 'Check that your OPENAI_API_KEY is valid and has image generation permissions'
    }

    if (lowerMessage.includes('model') || lowerMessage.includes('not found')) {
      return 'Verify your OpenAI organization has access to GPT Image 2.5 and has completed verification (https://platform.openai.com/settings/organization/general)'
    }

    if (lowerMessage.includes('forbidden') || lowerMessage.includes('permission')) {
      return 'Your OpenAI API key does not have permission for this operation or model'
    }

    return 'Check OpenAI API configuration and try again'
  }
}

export function createOpenAIImageClient(config: Config): Result<ImageClient, ImageAPIError> {
  try {
    const client = new OpenAI({
      apiKey: config.openaiApiKey,
      ...(config.openaiBaseUrl ? { baseURL: config.openaiBaseUrl } : {}),
    })
    return Ok(new OpenAIImageClientImpl(client, config.imageQuality))
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    return Err(
      new ImageAPIError(
        `Failed to initialize OpenAI image client: ${errorMessage}`,
        'Verify your OPENAI_API_KEY is valid and the openai package is properly installed'
      )
    )
  }
}
