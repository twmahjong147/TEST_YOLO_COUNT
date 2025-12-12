import CoreML
import UIKit

final class FastViTEmbedder: VisualEmbedder {
    private var model: MLModel?
    private let inputSize = CGSize(width: 256, height: 256)

    func loadModel() async throws {
        var modelURL: URL?

        // Try .mlmodelc (compiled model)
        if let compiledURL = Bundle.main.url(forResource: "FastViTMA36F16Headless", withExtension: "mlmodelc") {
            modelURL = compiledURL
        }
        // Try .mlpackage as fallback
        else if let packageURL = Bundle.main.url(forResource: "FastViTMA36F16Headless", withExtension: "mlpackage") {
            modelURL = packageURL
        }

        guard let modelURL = modelURL else {
            let bundlePath = Bundle.main.bundlePath
            throw ProcessingError.modelNotFound("FastViTMA36F16Headless model not found in bundle: \(bundlePath)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        do {
            model = try await MLModel.load(contentsOf: modelURL, configuration: config)
        } catch {
            throw ProcessingError.modelLoadFailed("FastViT: \(error.localizedDescription)")
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

        guard let outputValue = prediction.featureValue(for: "imageFeatures"),
              let multiArray = outputValue.multiArrayValue else {
            throw ProcessingError.predictionFailed("Failed to get embedding from FastViT")
        }

        var embedding: [Float] = []
        let count = multiArray.count
        embedding.reserveCapacity(count)

        for i in 0..<count {
            embedding.append(multiArray[i].floatValue)
        }

        // Ensure final vector is L2-normalized for similarity search
        return ImageProcessor.normalizeEmbedding(embedding)
    }
}
