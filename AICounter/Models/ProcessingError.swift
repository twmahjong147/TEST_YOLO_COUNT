import Foundation

enum ProcessingError: LocalizedError, Sendable {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case predictionFailed(String)
    case imageProcessingFailed(String)
    case noDetections
    case insufficientDetections
    
    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Model not found: \(name)"
        case .modelLoadFailed(let name):
            return "Failed to load model: \(name)"
        case .predictionFailed(let message):
            return "Prediction failed: \(message)"
        case .imageProcessingFailed(let message):
            return "Image processing failed: \(message)"
        case .noDetections:
            return "No objects detected in the image"
        case .insufficientDetections:
            return "Not enough objects detected for clustering"
        }
    }
}
