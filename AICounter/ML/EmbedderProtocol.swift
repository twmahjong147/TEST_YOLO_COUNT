import Foundation
import CoreGraphics

protocol VisualEmbedder {
    func loadModel() async throws
    func getEmbedding(for cropImage: CGImage) throws -> [Float]
}
