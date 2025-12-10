import Foundation

struct CountResult: Sendable {
    let count: Int
    let detections: [Detection]
    let largestClusterId: Int
    let clusterLabels: [Int]
    let processingTime: TimeInterval
    
    init(
        count: Int,
        detections: [Detection],
        largestClusterId: Int,
        clusterLabels: [Int],
        processingTime: TimeInterval
    ) {
        self.count = count
        self.detections = detections
        self.largestClusterId = largestClusterId
        self.clusterLabels = clusterLabels
        self.processingTime = processingTime
    }
}
