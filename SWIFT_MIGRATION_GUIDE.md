# Swift Migration Guide for Object Counting System

## Model Output Specifications

### YOLOX-S Output
- **Name**: `var_1430`
- **Shape**: `[1, 8400, 85]`
- **Format**: `[batch, num_predictions, features]`
- **Features**: `[x_center, y_center, width, height, objectness, class_scores...]`
  - Indices 0-3: Bounding box (x_center, y_center, width, height)
  - Index 4: Objectness score (confidence)
  - Indices 5-84: Class scores for 80 COCO classes

### TinyCLIP Vision Output
- **Name**: `var_651`
- **Shape**: `[1, 256]`
- **Format**: Normalized L2 embedding vector (256 dimensions)

## Swift Implementation Structure

### Project Structure
```
AICounter/
├── Models/
│   ├── yolox_s.mlpackage
│   └── tinyclip_vision.mlpackage
├── Core/
│   ├── YOLOXDetector.swift
│   ├── TinyCLIPEmbedder.swift
│   ├── SimilarityClusterer.swift
│   └── AICounter.swift
├── Utils/
│   ├── ImageProcessor.swift
│   ├── BoundingBox.swift
│   └── MathUtils.swift
└── Views/
    ├── ContentView.swift
    └── DetectionView.swift
```

## Core Swift Files

### 1. YOLOXDetector.swift

```swift
import CoreML
import Vision
import CoreImage

struct Detection {
    let bbox: CGRect
    let confidence: Float
    let classId: Int
}

class YOLOXDetector {
    private let model: VNCoreMLModel
    private let inputSize: CGSize = CGSize(width: 640, height: 640)
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
        let mlModel = try yolox_s(configuration: config).model
        self.model = try VNCoreMLModel(for: mlModel)
    }
    
    func detect(image: CGImage, confThreshold: Float = 0.25) throws -> [Detection] {
        var detections: [Detection] = []
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard error == nil,
                  let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let output = results.first?.featureValue.multiArrayValue else {
                return
            }
            
            // Parse YOLOX output: [1, 8400, 85]
            detections = self.parseYOLOXOutput(output, 
                                               imageSize: CGSize(width: image.width, height: image.height),
                                               confThreshold: confThreshold)
        }
        
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])
        
        return detections
    }
    
    private func parseYOLOXOutput(_ output: MLMultiArray, 
                                  imageSize: CGSize,
                                  confThreshold: Float) -> [Detection] {
        var detections: [Detection] = []
        let numPredictions = 8400
        let numFeatures = 85
        
        for i in 0..<numPredictions {
            // Get objectness score
            let objectness = output[[0, i, 4] as [NSNumber]].floatValue
            
            guard objectness > confThreshold else { continue }
            
            // Get bounding box (center format)
            let xCenter = output[[0, i, 0] as [NSNumber]].floatValue
            let yCenter = output[[0, i, 1] as [NSNumber]].floatValue
            let width = output[[0, i, 2] as [NSNumber]].floatValue
            let height = output[[0, i, 3] as [NSNumber]].floatValue
            
            // Convert to corner format and normalize
            let x = (xCenter - width / 2.0) / Float(inputSize.width) * Float(imageSize.width)
            let y = (yCenter - height / 2.0) / Float(inputSize.height) * Float(imageSize.height)
            let w = width / Float(inputSize.width) * Float(imageSize.width)
            let h = height / Float(inputSize.height) * Float(imageSize.height)
            
            // Find max class score
            var maxClassScore: Float = 0
            var maxClassId: Int = 0
            for j in 5..<numFeatures {
                let classScore = output[[0, i, j] as [NSNumber]].floatValue
                if classScore > maxClassScore {
                    maxClassScore = classScore
                    maxClassId = j - 5
                }
            }
            
            let confidence = objectness * maxClassScore
            guard confidence > confThreshold else { continue }
            
            let bbox = CGRect(x: CGFloat(x), y: CGFloat(y), 
                            width: CGFloat(w), height: CGFloat(h))
            
            detections.append(Detection(bbox: bbox, 
                                       confidence: confidence, 
                                       classId: maxClassId))
        }
        
        return detections
    }
}
```

### 2. TinyCLIPEmbedder.swift

```swift
import CoreML
import CoreImage
import Accelerate

class TinyCLIPEmbedder {
    private let model: tinyclip_vision
    private let inputSize: CGSize = CGSize(width: 224, height: 224)
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try tinyclip_vision(configuration: config)
    }
    
    func getEmbedding(for image: CGImage) throws -> [Float] {
        // Resize image to 224x224
        let resized = resizeImage(image, to: inputSize)
        
        // Convert to MLMultiArray format
        guard let pixelBuffer = createPixelBuffer(from: resized) else {
            throw NSError(domain: "TinyCLIP", code: -1, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }
        
        // Create input
        let input = tinyclip_visionInput(image: pixelBuffer)
        
        // Run inference
        let output = try model.prediction(input: input)
        
        // Extract embedding from var_651: [1, 256]
        return extractEmbedding(from: output.var_651)
    }
    
    private func extractEmbedding(from multiArray: MLMultiArray) -> [Float] {
        let count = 256
        var embedding = [Float](repeating: 0, count: count)
        
        for i in 0..<count {
            embedding[i] = multiArray[[0, i] as [NSNumber]].floatValue
        }
        
        return embedding
    }
    
    private func resizeImage(_ image: CGImage, to size: CGSize) -> CGImage {
        let context = CGContext(data: nil,
                               width: Int(size.width),
                               height: Int(size.height),
                               bitsPerComponent: 8,
                               bytesPerRow: 0,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: size))
        
        return context.makeImage()!
    }
    
    private func createPixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        let width = image.width
        let height = image.height
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                        width,
                                        height,
                                        kCVPixelFormatType_32BGRA,
                                        nil,
                                        &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                               width: width,
                               height: height,
                               bitsPerComponent: 8,
                               bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
}
```

### 3. SimilarityClusterer.swift

```swift
import Accelerate

class SimilarityClusterer {
    
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have same length")
        
        // For normalized vectors, cosine similarity = dot product
        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        
        return dotProduct
    }
    
    func cluster(embeddings: [[Float]], 
                threshold: Float = 0.80) -> [Int] {
        let n = embeddings.count
        guard n > 0 else { return [] }
        
        // Compute similarity matrix
        var similarities = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in i..<n {
                let sim = cosineSimilarity(embeddings[i], embeddings[j])
                similarities[i][j] = sim
                similarities[j][i] = sim
            }
        }
        
        // Agglomerative clustering
        var clusters = Array(0..<n) // Each item starts in its own cluster
        var clusterMap = (0..<n).map { [$0] } // Track items in each cluster
        
        // Merge clusters based on similarity
        var mergedClusters = Set<Int>()
        
        for i in 0..<n {
            if mergedClusters.contains(i) { continue }
            
            for j in (i+1)..<n {
                if mergedClusters.contains(j) { continue }
                
                if similarities[i][j] >= threshold {
                    // Merge cluster j into cluster i
                    let targetCluster = clusters[i]
                    for idx in clusterMap[j] {
                        clusters[idx] = targetCluster
                        clusterMap[i].append(idx)
                    }
                    clusterMap[j].removeAll()
                    mergedClusters.insert(j)
                }
            }
        }
        
        // Renumber clusters sequentially
        var uniqueClusters = Array(Set(clusters)).sorted()
        var clusterRemap = [Int: Int]()
        for (newId, oldId) in uniqueClusters.enumerated() {
            clusterRemap[oldId] = newId
        }
        
        return clusters.map { clusterRemap[$0]! }
    }
}
```

### 4. AICounter.swift

```swift
import CoreImage

struct CountResult {
    let count: Int
    let detections: [Detection]
    let largestClusterId: Int
    let clusterCounts: [Int: Int]
}

class AICounter {
    private let detector: YOLOXDetector
    private let embedder: TinyCLIPEmbedder
    private let clusterer: SimilarityClusterer
    
    init() throws {
        self.detector = try YOLOXDetector()
        self.embedder = try TinyCLIPEmbedder()
        self.clusterer = SimilarityClusterer()
    }
    
    func count(image: CGImage,
              confThreshold: Float = 0.25,
              similarityThreshold: Float = 0.80,
              outlierStdThreshold: Float = 3.0) throws -> CountResult {
        
        // Stage 1: Detect objects
        print("Stage 1: Object Detection")
        var detections = try detector.detect(image: image, confThreshold: confThreshold)
        print("  Detected \(detections.count) objects")
        
        guard !detections.isEmpty else {
            return CountResult(count: 0, detections: [], 
                             largestClusterId: -1, clusterCounts: [:])
        }
        
        // Stage 1.5: Filter size outliers
        if outlierStdThreshold > 0 {
            print("Stage 1.5: Size Outlier Filtering")
            detections = filterSizeOutliers(detections, stdThreshold: outlierStdThreshold)
            print("  Kept \(detections.count) after filtering")
        }
        
        // Stage 2: Extract embeddings
        print("Stage 2: Embedding Extraction")
        var embeddings: [[Float]] = []
        var validDetections: [Detection] = []
        
        for detection in detections {
            if let crop = cropImage(image, to: detection.bbox) {
                do {
                    let embedding = try embedder.getEmbedding(for: crop)
                    embeddings.append(embedding)
                    validDetections.append(detection)
                } catch {
                    print("  Warning: Failed to extract embedding for detection")
                }
            }
        }
        
        print("  Extracted \(embeddings.count) embeddings")
        
        // Stage 3: Cluster by similarity
        print("Stage 3: Similarity Clustering")
        let clusterLabels = clusterer.cluster(embeddings: embeddings, 
                                             threshold: similarityThreshold)
        
        // Count objects per cluster
        var clusterCounts: [Int: Int] = [:]
        for label in clusterLabels {
            clusterCounts[label, default: 0] += 1
        }
        
        print("  Found \(clusterCounts.count) clusters")
        
        // Find largest cluster
        let largestCluster = clusterCounts.max(by: { $0.value < $1.value })
        let finalCount = largestCluster?.value ?? 0
        let largestClusterId = largestCluster?.key ?? -1
        
        print("  Largest cluster: \(finalCount) objects")
        
        return CountResult(count: finalCount,
                         detections: validDetections,
                         largestClusterId: largestClusterId,
                         clusterCounts: clusterCounts)
    }
    
    private func filterSizeOutliers(_ detections: [Detection], 
                                    stdThreshold: Float) -> [Detection] {
        let areas = detections.map { Float($0.bbox.width * $0.bbox.height) }
        let median = calculateMedian(areas)
        let std = calculateStandardDeviation(areas, mean: median)
        
        let minArea = median - stdThreshold * std
        let maxArea = median + stdThreshold * std
        
        return detections.filter { detection in
            let area = Float(detection.bbox.width * detection.bbox.height)
            return area >= minArea && area <= maxArea
        }
    }
    
    private func calculateMedian(_ values: [Float]) -> Float {
        let sorted = values.sorted()
        let count = sorted.count
        
        if count % 2 == 0 {
            return (sorted[count/2 - 1] + sorted[count/2]) / 2.0
        } else {
            return sorted[count/2]
        }
    }
    
    private func calculateStandardDeviation(_ values: [Float], mean: Float) -> Float {
        let squaredDiffs = values.map { pow($0 - mean, 2) }
        let variance = squaredDiffs.reduce(0, +) / Float(values.count)
        return sqrt(variance)
    }
    
    private func cropImage(_ image: CGImage, to rect: CGRect) -> CGImage? {
        // Ensure rect is within image bounds
        let imageRect = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let clampedRect = rect.intersection(imageRect)
        
        guard !clampedRect.isEmpty else { return nil }
        
        return image.cropping(to: clampedRect)
    }
}
```

### 5. SwiftUI View Example

```swift
import SwiftUI

struct ContentView: View {
    @State private var image: UIImage?
    @State private var result: CountResult?
    @State private var isProcessing = false
    @State private var showImagePicker = false
    
    private let counter: AICounter
    
    init() {
        do {
            self.counter = try AICounter()
        } catch {
            fatalError("Failed to initialize AICounter: \(error)")
        }
    }
    
    var body: some View {
        VStack {
            if let image = image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 400)
            }
            
            if let result = result {
                Text("Count: \(result.count)")
                    .font(.largeTitle)
                    .padding()
                
                Text("Clusters: \(result.clusterCounts.count)")
                    .font(.subheadline)
            }
            
            Button("Select Image") {
                showImagePicker = true
            }
            .disabled(isProcessing)
            
            if let image = image {
                Button("Count Objects") {
                    processImage(image)
                }
                .disabled(isProcessing)
            }
        }
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(image: $image)
        }
    }
    
    private func processImage(_ image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        
        isProcessing = true
        
        Task {
            do {
                let result = try counter.count(image: cgImage)
                await MainActor.run {
                    self.result = result
                    self.isProcessing = false
                }
            } catch {
                print("Error counting objects: \(error)")
                await MainActor.run {
                    self.isProcessing = false
                }
            }
        }
    }
}
```

## Performance Optimization Tips

1. **Batch Processing**: Process multiple crops through TinyCLIP in batches
2. **GPU Acceleration**: Use `.computeUnits = .all` for best performance
3. **Memory Management**: Release large buffers promptly
4. **Background Processing**: Run inference on background threads
5. **Caching**: Cache model instances (don't recreate per request)

## Testing Checklist

- [ ] Load both Core ML models successfully
- [ ] YOLOX detection returns bounding boxes
- [ ] TinyCLIP embeddings are 256-dimensional normalized vectors
- [ ] Cosine similarity calculation is correct
- [ ] Clustering groups similar objects
- [ ] Size outlier filtering works
- [ ] Final count matches Python implementation
- [ ] Performance is acceptable (< 1s per image on modern devices)

## Migration Date
December 9, 2025
