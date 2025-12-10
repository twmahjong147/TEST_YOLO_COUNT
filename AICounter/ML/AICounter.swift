import UIKit
import CoreGraphics

@MainActor
final class AICounter {
    private let detector = YOLOXDetector()
    private let embedder = TinyCLIPEmbedder()
    
    private(set) var isLoaded = false
    
    func loadModels() async throws {
        try await detector.loadModel()
        try await embedder.loadModel()
        isLoaded = true
    }
    
    func count(
        image: CGImage,
        confidenceThreshold: Float = 0.001,
        similarityThreshold: Float = 0.80
    ) async throws -> CountResult {
        let startTime = Date()
        
        var detections = try detector.detect(
            image: image,
            confidenceThreshold: confidenceThreshold,
            nmsThreshold: 0.65
        )
        
        print("Stage 1: Detected \(detections.count) objects")
        
        guard !detections.isEmpty else {
            throw ProcessingError.noDetections
        }
        
        detections = filterSizeOutliers(detections, stdThreshold: 3.0)
        print("Stage 1.5: \(detections.count) after size filtering")
        
        detections = filterAspectRatioOutliers(detections, stdThreshold: 1.0)
        print("Stage 1.6: \(detections.count) after aspect ratio filtering")
        
        guard detections.count >= 2 else {
            throw ProcessingError.insufficientDetections
        }
        
        var embeddings: [[Float]] = []
        var validDetections: [Detection] = []
        
        for detection in detections {
            if let crop = ImageProcessor.cropImage(image, to: detection.bbox) {
                do {
                    let embedding = try embedder.getEmbedding(for: crop)
                    embeddings.append(embedding)
                    validDetections.append(detection)
                } catch {
                    continue
                }
            }
        }
        
        print("Stage 2: Extracted \(embeddings.count) embeddings")
        
        guard embeddings.count >= 2 else {
            throw ProcessingError.insufficientDetections
        }
        
        let clusterLabels = SimilarityClusterer.cluster(
            embeddings: embeddings,
            similarityThreshold: similarityThreshold
        )
        
        let clusterCounts = Dictionary(grouping: clusterLabels, by: { $0 })
            .mapValues { $0.count }
        
        print("Stage 3: Found \(clusterCounts.count) clusters")
        
        guard let largestCluster = clusterCounts.max(by: { $0.value < $1.value }) else {
            throw ProcessingError.insufficientDetections
        }
        
        // Build detections with cluster info (mirror Python behavior)
        var clusteredDetections: [Detection] = []
        for (idx, det) in validDetections.enumerated() {
            let clusterId = clusterLabels[idx]
            let isMain = (clusterId == largestCluster.key)
            // Set confidence to 1.0 if main cluster else 0.5 to mirror Python
            let conf: Float = isMain ? 1.0 : 0.5
            let newDet = Detection(bbox: det.bbox,
                                   confidence: conf,
                                   classId: det.classId,
                                   className: "cluster_\(clusterId)",
                                   objConf: det.objConf,
                                   clsConf: 1.0,
                                   clusterId: clusterId,
                                   isMainCluster: isMain)
            clusteredDetections.append(newDet)
            
            // Save debug crop image with overlay text to debug_outputs (best-effort)
            // if let cropCG = ImageProcessor.cropImage(image, to: det.bbox) {
            //     let ui = UIImage(cgImage: cropCG)
            //     let text = "Cluster \(clusterId)" + (isMain ? " (MAIN)" : "")
            //     UIGraphicsBeginImageContextWithOptions(ui.size, false, ui.scale)
            //     ui.draw(at: .zero)
            //     let attrs: [NSAttributedString.Key: Any] = [
            //         .font: UIFont.systemFont(ofSize: 12),
            //         .foregroundColor: UIColor.white,
            //         .backgroundColor: UIColor.black.withAlphaComponent(0.5)
            //     ]
            //     let textRect = CGRect(x: 4, y: 4, width: ui.size.width - 8, height: 20)
            //     text.draw(in: textRect, withAttributes: attrs)
            //     let annotated = UIGraphicsGetImageFromCurrentImageContext()
            //     UIGraphicsEndImageContext()
            //     if let annotated = annotated, let data = annotated.jpegData(compressionQuality: 0.9) {
            //         let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            //         if let docs = docs {
            //             let debugDir = docs.appendingPathComponent("debug_outputs")
            //             try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
            //             let fname = "crop_\(idx)_cluster_\(clusterId)_\(Int(Date().timeIntervalSince1970)).jpg"
            //             let url = debugDir.appendingPathComponent(fname)
            //             try? data.write(to: url)
            //         }
            //     }
            // }
        }
        
        let processingTime = Date().timeIntervalSince(startTime)
        
        return CountResult(
            count: largestCluster.value,
            detections: clusteredDetections,
            largestClusterId: largestCluster.key,
            clusterLabels: clusterLabels,
            processingTime: processingTime
        )
    }
    
    private func filterSizeOutliers(_ detections: [Detection], stdThreshold: Double) -> [Detection] {
        guard detections.count >= 3 else { return detections }
        
        let areas = detections.map { Double($0.area) }
        let medianArea = StatisticsHelper.median(areas)
        let stdArea = StatisticsHelper.standardDeviation(areas)
        
        let minArea = medianArea - (stdThreshold * stdArea)
        let maxArea = medianArea + (stdThreshold * stdArea)
        
        return detections.filter { detection in
            let area = Double(detection.area)
            return area >= minArea && area <= maxArea
        }
    }
    
    private func filterAspectRatioOutliers(_ detections: [Detection], stdThreshold: Double) -> [Detection] {
        guard detections.count >= 3 else { return detections }
        
        let ratios = detections.compactMap { detection -> Double? in
            guard detection.bbox.height > 0 else { return nil }
            return Double(detection.aspectRatio)
        }
        
        guard !ratios.isEmpty else { return detections }
        
        let medianRatio = StatisticsHelper.median(ratios)
        let stdRatio = StatisticsHelper.standardDeviation(ratios)
        
        guard stdRatio > 0 else { return detections }
        
        let minRatio = medianRatio - (stdThreshold * stdRatio)
        let maxRatio = medianRatio + (stdThreshold * stdRatio)
        
        return detections.filter { detection in
            guard detection.bbox.height > 0 else { return false }
            let ratio = Double(detection.aspectRatio)
            return ratio >= minRatio && ratio <= maxRatio
        }
    }
}
