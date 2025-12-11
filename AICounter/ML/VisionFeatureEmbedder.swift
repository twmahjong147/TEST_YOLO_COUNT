import Foundation
import Vision
import CoreImage

final class VisionFeatureEmbedder: VisualEmbedder {
    // Reuse a single CIContext to avoid per-call creation overhead
    private static let sharedCIContext: CIContext = {
        return CIContext(options: [CIContextOption.useSoftwareRenderer: true])
    }()

    func loadModel() async throws {
        // Touch the shared CIContext to create it and perform a tiny warm-up request
        _ = Self.sharedCIContext

        // Create a small warm-up image (16x16 grey) to initialize Vision/CoreImage subsystems
        let width = 16
        let height = 16
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
        if let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo) {
            ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            if let img = ctx.makeImage() {
                let request = VNGenerateImageFeaturePrintRequest()
                let handler = VNImageRequestHandler(cgImage: img, options: [VNImageOption.ciContext: Self.sharedCIContext])
                do {
                    try handler.perform([request])
                } catch {
                    // Best-effort warm-up; do not fail loadModel if warm-up fails
                    print("VisionFeatureEmbedder warm-up failed: \(error)")
                }
            }
        }
    }

    func getEmbedding(for cropImage: CGImage) throws -> [Float] {
        // Keep protocol-compatible entrypoint; default to non-verbose behavior
        // return try getEmbedding(for: cropImage, verbose: false)
        // Fast path: do not compute timings to avoid overhead
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cgImage: cropImage, options: [VNImageOption.ciContext: Self.sharedCIContext])
        do {
            try autoreleasepool {
                try handler.perform([request])
            }
        } catch {
            throw ProcessingError.predictionFailed("Vision featurePrint failed: \(error.localizedDescription)")
        }
                                                                                                                                                                                                  
        guard let obs = request.results?.first as? VNFeaturePrintObservation else {
            throw ProcessingError.predictionFailed("Failed to obtain VNFeaturePrintObservation")
        }
                                                                                                                                                                                                  
        let data = obs.data
            let floatCount = data.count / MemoryLayout<Float>.size
            var embedding = [Float](repeating: 0, count: floatCount)
   
            data.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) in
            let floatPtr = rawBuffer.bindMemory(to: Float32.self)
            for i in 0..<floatCount {
                embedding[i] = Float(floatPtr[i])
            }
        }
   
        return ImageProcessor.normalizeEmbedding(embedding)
    }
    
    // New overload with verbose flag. When verbose is true, compute and print detailed timings.
    func getEmbedding(for cropImage: CGImage, verbose: Bool) throws -> [Float] {
        let totalStart = CFAbsoluteTimeGetCurrent()

        let setupStart = CFAbsoluteTimeGetCurrent()
        let request = VNGenerateImageFeaturePrintRequest()
        // Use the shared CIContext to avoid recreate cost
        let handler = VNImageRequestHandler(cgImage: cropImage, options: [VNImageOption.ciContext: Self.sharedCIContext])
        let setupTime = CFAbsoluteTimeGetCurrent() - setupStart

        let performStart = CFAbsoluteTimeGetCurrent()
        do {
            try autoreleasepool {
                try handler.perform([request])
            }
        } catch {
            throw ProcessingError.predictionFailed("Vision featurePrint failed: \(error.localizedDescription)")
        }
        let performTime = CFAbsoluteTimeGetCurrent() - performStart

        let obsStart = CFAbsoluteTimeGetCurrent()
        guard let obs = request.results?.first as? VNFeaturePrintObservation else {
            throw ProcessingError.predictionFailed("Failed to obtain VNFeaturePrintObservation")
        }
        let obsTime = CFAbsoluteTimeGetCurrent() - obsStart

        let dataStart = CFAbsoluteTimeGetCurrent()
        let data = obs.data
        let dataTime = CFAbsoluteTimeGetCurrent() - dataStart

        let convertStart = CFAbsoluteTimeGetCurrent()
        let floatCount = data.count / MemoryLayout<Float>.size
        var embedding = [Float](repeating: 0, count: floatCount)

        data.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) in
            let floatPtr = rawBuffer.bindMemory(to: Float32.self)
            for i in 0..<floatCount {
                embedding[i] = Float(floatPtr[i])
            }
        }
        let convertTime = CFAbsoluteTimeGetCurrent() - convertStart

        let normStart = CFAbsoluteTimeGetCurrent()
        let normalized = ImageProcessor.normalizeEmbedding(embedding)
        let normTime = CFAbsoluteTimeGetCurrent() - normStart

        let totalTime = CFAbsoluteTimeGetCurrent() - totalStart

        if verbose {
            print("VisionFeatureEmbedder.getEmbedding timings (s): setup=\(setupTime) perform=\(performTime) obs=\(obsTime) dataCopy=\(dataTime) convert=\(convertTime) normalize=\(normTime) total=\(totalTime) count=\(floatCount)")
        }

        return normalized
    }
}
