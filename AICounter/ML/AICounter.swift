import UIKit
import CoreGraphics

@MainActor
final class AICounter {
    private let detector = YOLOXDetector()
    // Use Vision featurePrint by default for visual embeddings
    private let embedder: any VisualEmbedder = VisionFeatureEmbedder() // TinyCLIPEmbedder()
    
    private(set) var isLoaded = false
    
    func loadModels() async throws {
        try await detector.loadModel()
        try await embedder.loadModel()
        isLoaded = true
    }
    
    func count(
        image: CGImage,
        confidenceThreshold: Float = 0.001,
        nmsThreshold: Float = 0.65,
        similarityThreshold: Float = 0.80
    ) async throws -> CountResult {
        let startTime = Date()
        
        var detections = try detector.detect(
            image: image,
            confidenceThreshold: confidenceThreshold,
            nmsThreshold: nmsThreshold
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
        // var histograms: [[Float]] = []
        var validDetections: [Detection] = []
        
        for detection in detections {
            if let crop = ImageProcessor.cropImage(image, to: detection.bbox) {
                do {
                    let embedding = try embedder.getEmbedding(for: crop)
                    embeddings.append(embedding)
                    // compute color histogram for the crop (L2-normalised)
                    // let hist = ImageProcessor.colorHistogram(crop)
                    // histograms.append(hist)
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
            // colorHistograms: histograms,
            // histogramWeight: 0.20,
            similarityThreshold: similarityThreshold
        )
        
        let clusterCounts = Dictionary(grouping: clusterLabels, by: { $0 })
            .mapValues { $0.count }
        
        print("Stage 3: Found \(clusterCounts.count) clusters")
        
        guard let largestCluster = clusterCounts.max(by: { $0.value < $1.value }) else {
            throw ProcessingError.insufficientDetections
        }
        
        // Build detections with cluster info (mirror Python behavior)
        var mainClusteredDetections: [Detection] = []
        for (idx, det) in validDetections.enumerated() {
            let clusterId = clusterLabels[idx]
            if (clusterId != largestCluster.key) {
                continue
            }
            // Set confidence to 1.0 if main cluster else 0.5 to mirror Python
//            let conf: Float = 1.0
            let newDet = Detection(id: det.id,
                                   bbox: det.bbox,
                                   confidence: det.confidence,
                                   classId: det.classId,
                                   className: "cluster_\(clusterId)",
                                   objConf: det.objConf,
                                   clsConf: 1.0,
                                   clusterId: clusterId,
                                   isMainCluster: true)
            mainClusteredDetections.append(newDet)
            
            // Save debug crop image with overlay text to debug_outputs (best-effort)            
//            if let cropCG = ImageProcessor.cropImage(image, to: det.bbox) {
//                let ui = UIImage(cgImage: cropCG)
//                let text = "\(det.id)"
//                UIGraphicsBeginImageContextWithOptions(ui.size, false, ui.scale)
//                ui.draw(at: .zero)
//                let attrs: [NSAttributedString.Key: Any] = [
//                    .font: UIFont.systemFont(ofSize: 12),
//                    .foregroundColor: UIColor.white,
//                    .backgroundColor: UIColor.black.withAlphaComponent(0.5)
//                ]
//                let textRect = CGRect(x: 4, y: 4, width: ui.size.width - 8, height: 20)
//                text.draw(in: textRect, withAttributes: attrs)
//                let annotated = UIGraphicsGetImageFromCurrentImageContext()
//                UIGraphicsEndImageContext()
//                if let annotated = annotated, let data = annotated.jpegData(compressionQuality: 0.9) {
//                    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
//                    if let docs = docs {
//                        let debugDir = docs.appendingPathComponent("debug_outputs")
//                        try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
//                        let fname = "crop_\(idx)_id_\(det.id)_cluster_\(clusterId)_\(Int(Date().timeIntervalSince1970)).jpg"
//                        let url = debugDir.appendingPathComponent(fname)
//                        try? data.write(to: url)
//                    }
//                }
//            }
        }
        
        // Apply IoA (intersection over smaller-area) suppression within the main cluster
        // to remove fully-contained or near-duplicate boxes when only the main objects are needed.
        let ioaThreshold: Float = 0.9
        let sortedMain = mainClusteredDetections.sorted { $0.confidence > $1.confidence }
        var finalMain: [Detection] = []
        var suppressed = Set<Int>()
        for i in 0..<sortedMain.count {
            if suppressed.contains(i) { continue }
            finalMain.append(sortedMain[i])
            let areaI = sortedMain[i].area
            for j in (i + 1)..<sortedMain.count {
                if suppressed.contains(j) { continue }
                let inter = sortedMain[i].bbox.intersection(sortedMain[j].bbox)
                if inter.isNull { continue }
                let interArea = inter.width * inter.height
                let areaJ = sortedMain[j].area
                let ioa = Float(interArea / min(areaI, areaJ))
                if ioa > ioaThreshold {
                    suppressed.insert(j)
                }
            }
        }

        let processingTime = Date().timeIntervalSince(startTime)
        
        return CountResult(
            count: finalMain.count,
            detections: finalMain,
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
