import { existsSync } from 'node:fs'
import { extname } from 'node:path'
import type {
  AspectRatio,
  GenerateImageParams,
  ImageProvider,
  ImageQuality,
  ImageSize,
  OpenAIBackground,
  OpenAIInputFidelity,
  OpenAIInputImage,
  OpenAIModeration,
  OpenAIOutputFormat,
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
import type { Result } from '../types/result.js'
import { Err, Ok } from '../types/result.js'
import { InputValidationError } from '../utils/errors.js'
import { SUPPORTED_EXTENSIONS, SUPPORTED_MIME_TYPES } from '../utils/mimeUtils.js'

const PROMPT_MIN_LENGTH = 1
const PROMPT_MAX_LENGTH = 4000
export const MAX_IMAGE_SIZE = 10 * 1024 * 1024
const SUPPORTED_ASPECT_RATIOS = ASPECT_RATIO_VALUES
const SUPPORTED_QUALITY_VALUES = IMAGE_QUALITY_VALUES
const SUPPORTED_PROVIDER_VALUES = IMAGE_PROVIDER_VALUES

function formatFileSize(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

export function validatePrompt(prompt: unknown): Result<string, InputValidationError> {
  if (typeof prompt !== 'string') {
    return Err(
      new InputValidationError(
        'Prompt must be a non-empty string',
        'Please provide a descriptive prompt for image generation.'
      )
    )
  }
  if (prompt.length < PROMPT_MIN_LENGTH || prompt.length > PROMPT_MAX_LENGTH) {
    return Err(
      new InputValidationError(
        `Prompt must be between ${PROMPT_MIN_LENGTH} and ${PROMPT_MAX_LENGTH} characters. Current length: ${prompt.length}`,
        prompt.length === 0
          ? 'Please provide a descriptive prompt for image generation.'
          : `Please shorten your prompt by ${prompt.length - PROMPT_MAX_LENGTH} characters.`
      )
    )
  }

  if (prompt.trim().length === 0) {
    return Err(
      new InputValidationError(
        'Prompt must be a non-empty string',
        'Please provide a descriptive prompt for image generation.'
      )
    )
  }

  return Ok(prompt)
}

export function validateBase64Image(
  imageData?: string,
  mimeType?: string
): Result<Buffer | undefined, InputValidationError> {
  if (!imageData) {
    return Ok(undefined)
  }

  if (mimeType && !SUPPORTED_MIME_TYPES.includes(mimeType)) {
    return Err(
      new InputValidationError(
        `Unsupported MIME type: ${mimeType}. Supported types: ${SUPPORTED_MIME_TYPES.join(', ')}`,
        `Please provide an image with one of these MIME types: ${SUPPORTED_MIME_TYPES.join(', ')}`
      )
    )
  }

  const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/
  const cleanedData = imageData.replace(/^data:image\/[a-z]+;base64,/, '')

  if (!base64Regex.test(cleanedData)) {
    return Err(
      new InputValidationError(
        'Invalid base64 format',
        'Please provide a valid base64 encoded image string'
      )
    )
  }

  let buffer: Buffer
  try {
    buffer = Buffer.from(cleanedData, 'base64')

    if (buffer.length > MAX_IMAGE_SIZE) {
      const sizeInMB = formatFileSize(buffer.length)
      const limitInMB = formatFileSize(MAX_IMAGE_SIZE)
      return Err(
        new InputValidationError(
          `Image size exceeds ${limitInMB}MB limit. Current size: ${sizeInMB}MB`,
          `Please compress your image or reduce its resolution to stay below ${limitInMB}MB`
        )
      )
    }
  } catch (_error) {
    return Err(
      new InputValidationError(
        'Failed to decode base64 image',
        'Please ensure the image is properly base64 encoded'
      )
    )
  }

  return Ok(buffer)
}

function validateImagePath(imagePath?: string): Result<string | undefined, InputValidationError> {
  if (!imagePath) {
    return Ok(undefined)
  }

  if (!existsSync(imagePath)) {
    return Err(
      new InputValidationError(
        `Input image file not found: ${imagePath}`,
        'Please provide a valid absolute path to an existing image file'
      )
    )
  }

  const ext = extname(imagePath).toLowerCase()
  if (!SUPPORTED_EXTENSIONS.includes(ext)) {
    return Err(
      new InputValidationError(
        `Unsupported image format: ${ext}. Supported formats: ${SUPPORTED_EXTENSIONS.join(', ')}`,
        `Please provide an image with one of these extensions: ${SUPPORTED_EXTENSIONS.join(', ')}`
      )
    )
  }

  return Ok(imagePath)
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readOptionalString(input: Record<string, unknown>, field: string): string | undefined {
  const value = input[field]
  return typeof value === 'string' ? value : undefined
}

function readOptionalBoolean(input: Record<string, unknown>, field: string): boolean | undefined {
  const value = input[field]
  return typeof value === 'boolean' ? value : undefined
}

/** Fields whose only shared requirement is being a string when present. */
const OPTIONAL_STRING_FIELDS = [
  'fileName',
  'inputImagePath',
  'purpose',
  'inputImage',
  'inputImageMimeType',
  'aspectRatio',
  'imageSize',
  'quality',
  'provider',
] as const

/** Flags whose only shared requirement is being a boolean when present. */
const OPTIONAL_BOOLEAN_FIELDS = [
  {
    name: 'blendImages',
    suggestion:
      'Use true or false for blendImages parameter to enable/disable multi-image blending',
  },
  {
    name: 'maintainCharacterConsistency',
    suggestion:
      'Use true or false for maintainCharacterConsistency parameter to enable/disable character consistency',
  },
  {
    name: 'useWorldKnowledge',
    suggestion:
      'Use true or false for useWorldKnowledge parameter to enable/disable world knowledge integration',
  },
  {
    name: 'useGoogleSearch',
    suggestion:
      'Use true or false for useGoogleSearch parameter to enable/disable Google Search grounding',
  },
] as const

function checkOptionalStringFields(
  input: Record<string, unknown>
): InputValidationError | undefined {
  for (const field of OPTIONAL_STRING_FIELDS) {
    const value = input[field]
    if (value !== undefined && typeof value !== 'string') {
      return new InputValidationError(
        `${field} must be a string`,
        `Provide a string for ${field} or omit it`
      )
    }
  }
  return undefined
}

function checkOptionalBooleanFields(
  input: Record<string, unknown>
): InputValidationError | undefined {
  for (const { name, suggestion } of OPTIONAL_BOOLEAN_FIELDS) {
    const value = input[name]
    if (value !== undefined && typeof value !== 'boolean') {
      return new InputValidationError(`${name} must be a boolean value`, suggestion)
    }
  }
  return undefined
}

function isMemberOf<T extends string>(values: readonly T[], value: string): value is T {
  return values.some((candidate) => candidate === value)
}

interface EnumFields {
  aspectRatio?: AspectRatio
  imageSize?: ImageSize
  quality?: ImageQuality
  provider?: ImageProvider
}

/**
 * Validate the four enum-valued fields. Each keeps its own message; only the
 * membership check is shared.
 */
function validateEnumFields(
  input: Record<string, unknown>
): Result<EnumFields, InputValidationError> {
  const fields: EnumFields = {}

  const imageSize = readOptionalString(input, 'imageSize')
  if (imageSize !== undefined) {
    if (!isMemberOf(IMAGE_SIZE_VALUES, imageSize)) {
      return Err(
        new InputValidationError(
          `Invalid image size: ${imageSize}`,
          `Use one of: ${IMAGE_SIZE_VALUES.join(', ')}`
        )
      )
    }
    fields.imageSize = imageSize
  }

  const aspectRatio = readOptionalString(input, 'aspectRatio')
  if (aspectRatio !== undefined) {
    if (!isMemberOf(SUPPORTED_ASPECT_RATIOS, aspectRatio)) {
      return Err(
        new InputValidationError(
          `Invalid aspect ratio: ${aspectRatio}. Supported values: ${SUPPORTED_ASPECT_RATIOS.join(', ')}`,
          `Please use one of the supported aspect ratios: ${SUPPORTED_ASPECT_RATIOS.join(', ')}`
        )
      )
    }
    fields.aspectRatio = aspectRatio
  }

  const quality = readOptionalString(input, 'quality')
  if (quality !== undefined) {
    if (!isMemberOf(SUPPORTED_QUALITY_VALUES, quality)) {
      return Err(
        new InputValidationError(
          `Invalid quality value: "${quality}". Supported values: ${SUPPORTED_QUALITY_VALUES.join(', ')}`,
          `Please use one of the supported quality values: ${SUPPORTED_QUALITY_VALUES.join(', ')}`
        )
      )
    }
    fields.quality = quality
  }

  const provider = readOptionalString(input, 'provider')
  if (provider !== undefined) {
    if (!isMemberOf(SUPPORTED_PROVIDER_VALUES, provider)) {
      return Err(
        new InputValidationError(
          `Invalid provider value: "${provider}". Supported values: ${SUPPORTED_PROVIDER_VALUES.join(', ')}`,
          `Please use one of the supported providers: ${SUPPORTED_PROVIDER_VALUES.join(', ')}`
        )
      )
    }
    fields.provider = provider
  }

  return Ok(fields)
}

function assignOptional<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: T[K] | undefined
): void {
  if (value !== undefined) {
    target[key] = value
  }
}

export function validateGenerateImageParams(
  input: unknown
): Result<GenerateImageParams, InputValidationError> {
  if (!isPlainObject(input)) {
    return Err(
      new InputValidationError(
        'Tool arguments must be an object',
        'Provide an object containing a prompt and optional image parameters'
      )
    )
  }

  const stringFieldError = checkOptionalStringFields(input)
  if (stringFieldError) {
    return Err(stringFieldError)
  }

  const promptResult = validatePrompt(input['prompt'])
  if (!promptResult.success) {
    return Err(promptResult.error)
  }

  const inputImagePath = readOptionalString(input, 'inputImagePath')
  const imagePathResult = validateImagePath(inputImagePath)
  if (!imagePathResult.success) {
    return Err(imagePathResult.error)
  }

  const booleanFieldError = checkOptionalBooleanFields(input)
  if (booleanFieldError) {
    return Err(booleanFieldError)
  }

  const inputImage = readOptionalString(input, 'inputImage')
  const inputImageMimeType = readOptionalString(input, 'inputImageMimeType')
  if (inputImage || inputImageMimeType) {
    const imageResult = validateBase64Image(inputImage, inputImageMimeType)
    if (!imageResult.success) {
      return Err(imageResult.error)
    }
  }

  const enumFieldsResult = validateEnumFields(input)
  if (!enumFieldsResult.success) {
    return Err(enumFieldsResult.error)
  }

  const openAIOptions = readOpenAIImageOptions(input)
  if (!openAIOptions.success) {
    return Err(openAIOptions.error)
  }

  const params: GenerateImageParams = { prompt: promptResult.data, ...enumFieldsResult.data }
  assignOptional(params, 'fileName', readOptionalString(input, 'fileName'))
  assignOptional(params, 'inputImagePath', inputImagePath)
  assignOptional(params, 'inputImage', inputImage)
  assignOptional(params, 'inputImageMimeType', inputImageMimeType)
  assignOptional(params, 'purpose', readOptionalString(input, 'purpose'))
  for (const { name } of OPTIONAL_BOOLEAN_FIELDS) {
    assignOptional(params, name, readOptionalBoolean(input, name))
  }
  assignOptional(params, 'background', openAIOptions.data.background)
  assignOptional(params, 'inputFidelity', openAIOptions.data.inputFidelity)
  assignOptional(params, 'moderation', openAIOptions.data.moderation)
  assignOptional(params, 'outputCompression', openAIOptions.data.outputCompression)
  assignOptional(params, 'imageCount', openAIOptions.data.imageCount)
  assignOptional(params, 'outputFormat', openAIOptions.data.outputFormat)
  assignOptional(params, 'maskImage', openAIOptions.data.maskImage)
  assignOptional(params, 'maskImagePath', openAIOptions.data.maskImagePath)
  assignOptional(params, 'inputImages', openAIOptions.data.inputImages)

  return Ok(params)
}

const MAX_OPENAI_INPUT_IMAGES = 16

function readEnumOption<T extends string>(
  input: Record<string, unknown>,
  field: string,
  allowed: readonly T[]
): Result<T | undefined, InputValidationError> {
  const value = input[field]
  if (value === undefined) {
    return Ok(undefined)
  }
  if (typeof value !== 'string' || !isMemberOf(allowed, value)) {
    return Err(
      new InputValidationError(
        `Invalid ${field}: ${String(value)}`,
        `Use one of: ${allowed.join(', ')}`
      )
    )
  }
  return Ok(value)
}

function readBoundedInteger(
  input: Record<string, unknown>,
  field: string,
  min: number,
  max: number
): Result<number | undefined, InputValidationError> {
  const value = input[field]
  if (value === undefined) {
    return Ok(undefined)
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    return Err(
      new InputValidationError(
        `${field} must be an integer from ${min} to ${max}`,
        `Set ${field} between ${min} and ${max}, or omit it`
      )
    )
  }
  return Ok(value)
}

function readOpenAIImageOptions(
  input: Record<string, unknown>
): Result<
  Pick<
    GenerateImageParams,
    | 'background'
    | 'inputFidelity'
    | 'moderation'
    | 'outputCompression'
    | 'imageCount'
    | 'outputFormat'
    | 'maskImage'
    | 'maskImagePath'
    | 'inputImages'
  >,
  InputValidationError
> {
  const background = readEnumOption<OpenAIBackground>(input, 'background', OPENAI_BACKGROUND_VALUES)
  if (!background.success) {
    return background
  }
  const inputFidelity = readEnumOption<OpenAIInputFidelity>(
    input,
    'inputFidelity',
    OPENAI_INPUT_FIDELITY_VALUES
  )
  if (!inputFidelity.success) {
    return inputFidelity
  }
  const moderation = readEnumOption<OpenAIModeration>(input, 'moderation', OPENAI_MODERATION_VALUES)
  if (!moderation.success) {
    return moderation
  }
  const outputFormat = readEnumOption<OpenAIOutputFormat>(
    input,
    'outputFormat',
    OPENAI_OUTPUT_FORMAT_VALUES
  )
  if (!outputFormat.success) {
    return outputFormat
  }
  const outputCompression = readBoundedInteger(input, 'outputCompression', 0, 100)
  if (!outputCompression.success) {
    return outputCompression
  }
  const imageCount = readBoundedInteger(input, 'imageCount', 1, 10)
  if (!imageCount.success) {
    return imageCount
  }

  const mask = readMaskOptions(input)
  if (!mask.success) {
    return mask
  }

  const inputImages = readInputImages(input)
  if (!inputImages.success) {
    return inputImages
  }

  return Ok({
    ...(background.data !== undefined && { background: background.data }),
    ...(inputFidelity.data !== undefined && { inputFidelity: inputFidelity.data }),
    ...(moderation.data !== undefined && { moderation: moderation.data }),
    ...(outputFormat.data !== undefined && { outputFormat: outputFormat.data }),
    ...(outputCompression.data !== undefined && { outputCompression: outputCompression.data }),
    ...(imageCount.data !== undefined && { imageCount: imageCount.data }),
    ...mask.data,
    ...(inputImages.data !== undefined && { inputImages: inputImages.data }),
  })
}

function readMaskOptions(
  input: Record<string, unknown>
): Result<Pick<GenerateImageParams, 'maskImage' | 'maskImagePath'>, InputValidationError> {
  const maskImage = readOptionalString(input, 'maskImage')
  const maskImagePath = readOptionalString(input, 'maskImagePath')
  if (input['maskImage'] !== undefined && typeof input['maskImage'] !== 'string') {
    return Err(new InputValidationError('maskImage must be a string', 'Provide a base64 PNG mask'))
  }
  if (input['maskImagePath'] !== undefined && typeof input['maskImagePath'] !== 'string') {
    return Err(
      new InputValidationError('maskImagePath must be a string', 'Provide a path to a PNG mask')
    )
  }
  if (maskImage) {
    const maskResult = validateBase64Image(maskImage, 'image/png')
    if (!maskResult.success) {
      return Err(maskResult.error)
    }
  }
  if (maskImagePath) {
    const maskPathResult = validateImagePath(maskImagePath)
    if (!maskPathResult.success) {
      return Err(maskPathResult.error)
    }
  }

  return Ok({
    ...(maskImage !== undefined && { maskImage }),
    ...(maskImagePath !== undefined && { maskImagePath }),
  })
}

function readInputImages(
  input: Record<string, unknown>
): Result<OpenAIInputImage[] | undefined, InputValidationError> {
  const value = input['inputImages']
  if (value === undefined) {
    return Ok(undefined)
  }
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_OPENAI_INPUT_IMAGES) {
    return Err(
      new InputValidationError(
        `inputImages must contain 1 to ${MAX_OPENAI_INPUT_IMAGES} images`,
        'Provide an array of { data, mimeType } image objects'
      )
    )
  }

  const images: OpenAIInputImage[] = []
  for (const item of value) {
    if (!isPlainObject(item) || typeof item['data'] !== 'string') {
      return Err(
        new InputValidationError(
          'Each inputImages item must include base64 data',
          'Use { data: "<base64>", mimeType: "image/png" }'
        )
      )
    }
    const mimeType = typeof item['mimeType'] === 'string' ? item['mimeType'] : undefined
    const decoded = validateBase64Image(item['data'], mimeType)
    if (!decoded.success) {
      return Err(decoded.error)
    }
    if (!decoded.data) {
      return Err(
        new InputValidationError(
          'Each inputImages item must include base64 data',
          'Use { data: "<base64>", mimeType: "image/png" }'
        )
      )
    }
    images.push({
      data: decoded.data.toString('base64'),
      ...(mimeType !== undefined && { mimeType }),
    })
  }
  return Ok(images)
}
