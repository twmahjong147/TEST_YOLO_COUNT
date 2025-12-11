import Foundation
import CoreGraphics

@MainActor
protocol VisualEmbedder {
    func loadModel() async throws
    func getEmbedding(for cropImage: CGImage) throws -> [Float]
}
