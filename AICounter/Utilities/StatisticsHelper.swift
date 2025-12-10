import Foundation
import Accelerate

enum StatisticsHelper {
    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let count = sorted.count
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            return sorted[count / 2]
        }
    }
    
    static func standardDeviation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        
        let mean = values.reduce(0, +) / Double(values.count)
        let squaredDiffs = values.map { pow($0 - mean, 2) }
        let variance = squaredDiffs.reduce(0, +) / Double(values.count)
        
        return sqrt(variance)
    }
    
    static func cosineSimilarity(_ vec1: [Float], _ vec2: [Float]) -> Float {
        guard vec1.count == vec2.count else { return 0 }
        
        var dotProduct: Float = 0
        var mag1: Float = 0
        var mag2: Float = 0
        
        vDSP_dotpr(vec1, 1, vec2, 1, &dotProduct, vDSP_Length(vec1.count))
        
        var vec1Copy = vec1
        var vec2Copy = vec2
        vDSP_svesq(&vec1Copy, 1, &mag1, vDSP_Length(vec1.count))
        vDSP_svesq(&vec2Copy, 1, &mag2, vDSP_Length(vec2.count))
        
        mag1 = sqrt(mag1)
        mag2 = sqrt(mag2)
        
        guard mag1 > 0, mag2 > 0 else { return 0 }
        
        return dotProduct / (mag1 * mag2)
    }
}
