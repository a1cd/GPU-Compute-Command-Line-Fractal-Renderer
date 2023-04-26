import Foundation
import MetalKit

// Define the size of the fractal image
let imagemul = 1
let imageSize = CGSize(width: 1024*imagemul, height: 1024 * imagemul)

// Create a Metal device and command queue
guard let device = MTLCreateSystemDefaultDevice(),
      let commandQueue = device.makeCommandQueue()
else {
    fatalError("Failed to create Metal device or command queue")
}

// Load the flame fractal compute kernel from the .metal file
let library = device.makeDefaultLibrary()
let kernelFunction = library!.makeFunction(name: "flameFractal")!


// capture manager
let sharedCapturer = MTLCaptureManager.shared()
let customScope = sharedCapturer.makeCaptureScope(device: device)
// Add a label if you want to capture it from XCode's debug bar
customScope.label = "Pls debug me"
// If you want to set this scope as the default debug scope, assign it to MTLCaptureManager's defaultCaptureScope
sharedCapturer.defaultCaptureScope = customScope

let numThreads = imageSize.width * imageSize.height

let fractalOutputBuffer = device.makeBuffer(length: Int(numThreads) * MemoryLayout<UInt8>.size * 4, options: [])
let params: [Float] = [
    // Probability weights
    0.1, 0.1, 0.1, 0.1, 0.6,
    // Affine transform matrices
    0.85,  0.04, -0.04,  0.85,  0.00,  1.60,
    0.20, -0.26,  0.23,  0.22,  0.00,  1.60,
   -0.15,  0.28,  0.26,  0.24,  0.00,  0.44,
    0.00,  0.00,  0.00,  0.16,  0.00,  0.00,
    // Probability thresholds
    0.01, 0.08, 0.15, 0.80, 1.00
]

// Get a pointer to the output buffer contents
let fractalOutputBufferPointer = fractalOutputBuffer?.contents().bindMemory(to: Float.self, capacity: Int(numThreads * 4))

// Set up the compute pipeline and encoder
let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
pipelineState.makeComputePipelineStateWithAdditionalBinaryFunctions(functions: <#T##[MTLFunction]#>)
customScope.begin()
let commandBuffer = commandQueue.makeCommandBuffer()!
let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
computeEncoder.setComputePipelineState(pipelineState)
computeEncoder.setBuffer(fractalOutputBuffer, offset: 0, index: 0)
computeEncoder.setBytes(params, length: MemoryLayout<Float>.stride * params.count, index: 1)

let rngStatesBufferSize = MemoryLayout<Float>.size * Int(numThreads) * 2
guard let rngStatesBuffer = device.makeBuffer(length: rngStatesBufferSize, options: []) else {
    fatalError("Failed to create RNG states buffer")
}
let rngStatesBufferPointer = rngStatesBuffer.contents().bindMemory(to: Float.self, capacity: Int(rngStatesBufferSize))

// Initialize the RNG states buffer
for i in 0..<(Int(numThreads) * 2) {
    rngStatesBufferPointer[i] = Float(arc4random()) / Float(UINT32_MAX)
}

// Set the buffer for rngStates at index 2
computeEncoder.setBuffer(rngStatesBuffer, offset: 0, index: 2)

let w = pipelineState.threadExecutionWidth
let h = pipelineState.maxTotalThreadsPerThreadgroup / w
print(w)
print(h)
let maxThreadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
let numThreadgroups = MTLSize(
    width: Int(imageSize.width)/maxThreadsPerThreadgroup.width,//16,//(Int(numThreads) + maxThreadsPerThreadgroup.width - 1) / maxThreadsPerThreadgroup.width,
    height: Int(imageSize.height)/maxThreadsPerThreadgroup.height,
    depth: 1
)
var imageSizeMTL = MTLSizeMake(Int(imageSize.width), Int(imageSize.width), 1)
var threadGroupSizeMTL = maxThreadsPerThreadgroup

computeEncoder.setBytes(&imageSizeMTL, length: MemoryLayout<MTLSize>.stride, index: 3)
computeEncoder.setBytes(&threadGroupSizeMTL, length: MemoryLayout<MTLSize>.stride, index: 4)

var texture = device.makeSharedTexture(descriptor: MTLTextureDescriptor())
computeEncoder.setTexture(texture, index: 0)
// Dispatch the compute kernel
computeEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: maxThreadsPerThreadgroup)

// End the compute encoder and command buffer
computeEncoder.endEncoding()
commandBuffer.commit()
customScope.end()

// Convert the fractal data to an image
let imageByteCount = Int(imageSize.width * imageSize.height) * MemoryLayout<UInt8>.stride * 4
var imageBytes = [UInt8](repeating: 0, count: Int(imageSize.width * imageSize.height * 4))
memcpy(&imageBytes, fractalOutputBufferPointer, imageByteCount)
var imageBytesUInt8 = imageBytes//imageBytes.map { UInt8($0 * 255.0) }

let colorSpace = CGColorSpaceCreateDeviceRGB()
let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
let renderingIntent = CGColorRenderingIntent.defaultIntent//defaultIntent

guard let data = CFDataCreate(kCFAllocatorDefault, imageBytesUInt8, imageByteCount) else {
    fatalError("Failed to create CFData")
}

let provider = CGDataProvider(data: data)!
let cgImage = CGImage(
    width: Int(imageSize.width),
    height: Int(imageSize.height),
    bitsPerComponent: 8,
    bitsPerPixel: 8*4,
    bytesPerRow: Int(imageSize.width) * MemoryLayout<UInt8>.stride * 4,
    space: colorSpace,
    bitmapInfo: bitmapInfo,
    provider: provider,
    decode: nil,
    shouldInterpolate: true,
    intent: renderingIntent
)


// Save the image to disk
var temDir = NSTemporaryDirectory()
let url = URL(fileURLWithPath: "./flame_fractal.png", relativeTo: NSURL(fileURLWithPath: temDir, isDirectory: true) as URL)
guard let destination = CGImageDestinationCreateWithURL(url as CFURL, kUTTypePNG, 1, nil) else {
    fatalError("Failed to create image destination")
}
CGImageDestinationAddImage(destination, cgImage!, nil)
CGImageDestinationFinalize(destination)
print(url)
