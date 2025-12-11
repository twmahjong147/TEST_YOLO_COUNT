import Foundation
import Vision
import CoreImage

@MainActor
final class VisionFeatureEmbedder: VisualEmbedder {
    func loadModel() async throws {
        // Vision's feature print runs on device APIs; no explicit model load required.
    }

    func getEmbedding(for cropImage: CGImage) throws -> [Float] {
        let request = VNGenerateImageFeaturePrintRequest()
        // Use a software CIContext to avoid creating GPU/Espresso contexts on some devices/simulators
        let ciContext = CIContext(options: [CIContextOption.useSoftwareRenderer: true])
        let handler = VNImageRequestHandler(cgImage: cropImage, options: [VNImageOption.ciContext: ciContext])

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
}
