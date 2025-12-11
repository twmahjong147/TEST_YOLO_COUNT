import CoreML
import UIKit
import Vision

@MainActor
final class YOLOXDetector {
    private var model: MLModel?
    private let inputSize = CGSize(width: 640, height: 640)
    
    func loadModel() async throws {
        // Look for compiled model first
        var modelURL: URL?

        // Try .mlmodelc (compiled model)
        if let compiledURL = Bundle.main.url(forResource: "yolox_l", withExtension: "mlmodelc") {
            modelURL = compiledURL
        }
        // Try .mlpackage as fallback
        else if let packageURL = Bundle.main.url(forResource: "yolox_l", withExtension: "mlpackage")
        {
            modelURL = packageURL
        }
        // Try direct resource lookup
        else if let resourceURL = Bundle.main.resourceURL?.appendingPathComponent(
            "yolox_l.mlmodelc")
        {
            if FileManager.default.fileExists(atPath: resourceURL.path) {
                modelURL = resourceURL
            }
        }

        guard let modelURL = modelURL else {
            let bundlePath = Bundle.main.bundlePath
            throw ProcessingError.modelNotFound("yolox_l model not found in bundle: \(bundlePath)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        do {
            model = try await MLModel.load(contentsOf: modelURL, configuration: config)
        } catch {
            throw ProcessingError.modelLoadFailed("yolox_l: \(error.localizedDescription)")
        }
    }

    func detect(
        image: CGImage,
        confidenceThreshold: Float = 0.001,
        nmsThreshold: Float = 0.65
    ) throws -> [Detection] {
        guard let model = model else {
            throw ProcessingError.modelLoadFailed("Model not loaded")
        }

        guard let (chwBuffer, scale) = ImageProcessor.preProc(image, to: inputSize) else {
            throw ProcessingError.imageProcessingFailed("Failed to prepare input")
        }

        // Convert CHW buffer to MLMultiArray for Core ML input
        guard let inputArray = try? MLMultiArray(shape: [1, 3, NSNumber(value: Int(inputSize.height)), NSNumber(value: Int(inputSize.width))], dataType: .float32) else {
            throw ProcessingError.imageProcessingFailed("Failed to create MLMultiArray")
        }
        for (i, v) in chwBuffer.enumerated() {
            inputArray[i] = NSNumber(value: v)
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: inputArray)
        ])

        let prediction = try model.prediction(from: input)

        // Extract output multiarray and run postprocess (matches official YOLOX postprocess)
        guard let outputValue = prediction.featureValue(for: "var_2188"), //"var_1430"),
              let multiArray = outputValue.multiArrayValue
        else {
            throw ProcessingError.predictionFailed("Failed to get output from model")
        }

        let detections = postProcess(multiArray: multiArray,
                                     imageSize: CGSize(width: image.width, height: image.height),
                                     numClasses: 80,
                                     confThreshold: confidenceThreshold,
                                     nmsThreshold: nmsThreshold,
                                     classAgnostic: true)

        // Scale boxes back to original image coordinates by dividing by the preprocessing scale (like Python's output[:, :4] /= ratio)
        let scaledDetections = detections.map { det -> Detection in
            let b = det.bbox
            let scaledBBox = CGRect(x: b.origin.x / scale,
                                    y: b.origin.y / scale,
                                    width: b.width / scale,
                                    height: b.height / scale)
            return Detection(id: det.id, bbox: scaledBBox, confidence: det.confidence, classId: det.classId)
        }

        return scaledDetections
    }

    private func applyNMS(detections: [Detection], iouThreshold: Float) -> [Detection] {
        guard detections.count > 1 else { return detections }

        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var keep: [Detection] = []
        var suppressed = Set<Int>()
        
        for (i, detection) in sorted.enumerated() {
            if suppressed.contains(i) { continue }

            keep.append(detection)

            for j in (i + 1)..<sorted.count {
                if suppressed.contains(j) { continue }

                let iou = calculateIoU(detection.bbox, sorted[j].bbox)
                if iou > iouThreshold {
                    suppressed.insert(j)
                }
            }
        }

        return keep
    }

    // Swift port of official_yolox.postprocess
    private func postProcess(multiArray: MLMultiArray,
                             imageSize: CGSize,
                             numClasses: Int,
                             confThreshold: Float = 0.001,
                             nmsThreshold: Float = 0.65,
                             classAgnostic: Bool = false) -> [Detection] {
        // multiArray shape: [1, N, 5+numClasses]
        let numAnchors = multiArray.shape[1].intValue
        let numValues = multiArray.shape[2].intValue

        var detections: [Detection] = []

        // Note: do not rescale coordinates here to match Python postprocess which returns boxes in model/input image coordinates

        for i in 0..<numAnchors {
            let objectness = multiArray[[0, NSNumber(value: i), 4]].floatValue

            // compute class conf and id
            var maxClassScore: Float = 0
            var maxClassId: Int = 0
            for c in 5..<(5 + numClasses) {
                let score = multiArray[[0, NSNumber(value: i), NSNumber(value: c)]].floatValue
                if score > maxClassScore {
                    maxClassScore = score
                    maxClassId = c - 5
                }
            }

            let confidence = objectness * maxClassScore
            if confidence < confThreshold { continue }

            // Keep coordinates in model/input scale (no multiplication by image/input size ratio)
            let xCenter = multiArray[[0, NSNumber(value: i), 0]].floatValue
            let yCenter = multiArray[[0, NSNumber(value: i), 1]].floatValue
            let w = multiArray[[0, NSNumber(value: i), 2]].floatValue
            let h = multiArray[[0, NSNumber(value: i), 3]].floatValue

            let x1 = xCenter - w / 2.0
            let y1 = yCenter - h / 2.0

            let bbox = CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(w), height: CGFloat(h))
            
            detections.append(Detection(id: i, bbox: bbox, confidence: confidence, classId: maxClassId))
        }
        
        // If no detections, return empty
        if detections.isEmpty { return [] }

        // Perform NMS. If classAgnostic == false, perform per-class NMS (batched_nms)
        if classAgnostic {
            return applyNMS(detections: detections, iouThreshold: nmsThreshold)
        } else {
            var final: [Detection] = []
            // group by class
            let grouped = Dictionary(grouping: detections, by: { $0.classId })
            for (_, dets) in grouped {
                let kept = applyNMS(detections: dets, iouThreshold: nmsThreshold)
                final.append(contentsOf: kept)
            }
            return final
        }
    }

    private func calculateIoU(_ box1: CGRect, _ box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)

        guard !intersection.isNull else { return 0 }

        let intersectionArea = intersection.width * intersection.height
        let union = box1.width * box1.height + box2.width * box2.height - intersectionArea

        guard union > 0 else { return 0 }

        return Float(intersectionArea / union)
    }
}
