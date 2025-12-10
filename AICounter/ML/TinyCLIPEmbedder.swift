import CoreML
import UIKit

@MainActor
final class TinyCLIPEmbedder {
    private var model: MLModel?
    private let inputSize = CGSize(width: 224, height: 224)
    
    func loadModel() async throws {
        // Look for compiled model first
        var modelURL: URL?
        
        // Try .mlmodelc (compiled model)
        if let compiledURL = Bundle.main.url(forResource: "tinyclip_vision", withExtension: "mlmodelc") {
            modelURL = compiledURL
        }
        // Try .mlpackage as fallback
        else if let packageURL = Bundle.main.url(forResource: "tinyclip_vision", withExtension: "mlpackage") {
            modelURL = packageURL
        }
        // Try direct resource lookup
        else if let resourceURL = Bundle.main.resourceURL?.appendingPathComponent("tinyclip_vision.mlmodelc") {
            if FileManager.default.fileExists(atPath: resourceURL.path) {
                modelURL = resourceURL
            }
        }
        
        guard let modelURL = modelURL else {
            let bundlePath = Bundle.main.bundlePath
            throw ProcessingError.modelNotFound("tinyclip_vision model not found in bundle: \(bundlePath)")
        }
        
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        do {
            model = try await MLModel.load(contentsOf: modelURL, configuration: config)
        } catch {
            throw ProcessingError.modelLoadFailed("tinyclip_vision: \(error.localizedDescription)")
        }
    }
    
    func getEmbedding(for cropImage: CGImage) throws -> [Float] {
        guard let model = model else {
            throw ProcessingError.modelLoadFailed("Model not loaded")
        }
        
        guard let resizedImage = ImageProcessor.resizeImage(cropImage, to: inputSize),
              let pixelBuffer = ImageProcessor.createPixelBuffer(
                from: resizedImage,
                width: Int(inputSize.width),
                height: Int(inputSize.height)
              ) else {
            throw ProcessingError.imageProcessingFailed("Failed to prepare crop for embedding")
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)])
        
        let prediction = try model.prediction(from: input)
        
        guard let outputValue = prediction.featureValue(for: "var_651"),
              let multiArray = outputValue.multiArrayValue else {
            throw ProcessingError.predictionFailed("Failed to get embedding from TinyCLIP")
        }
        
        var embedding: [Float] = []
        let count = multiArray.count
        
        for i in 0..<count {
            embedding.append(multiArray[i].floatValue)
        }
        
        return ImageProcessor.normalizeEmbedding(embedding)
    }
}
