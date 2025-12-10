import Foundation
import CoreGraphics

struct Detection: Identifiable, Sendable {
    let id = UUID()
    let bbox: CGRect
    let confidence: Float
    let classId: Int

    // Additional fields to mirror Python detection dict
    var className: String?            // e.g., "cluster_0"
    var objConf: Float?               // objectness confidence
    var clsConf: Float?               // class confidence
    var area: CGFloat {               // area in pixels
        bbox.width * bbox.height
    }
    var clusterId: Int?               // cluster assignment
    var isMainCluster: Bool = false

    init(bbox: CGRect,
         confidence: Float,
         classId: Int = 0,
         className: String? = nil,
         objConf: Float? = nil,
         clsConf: Float? = nil,
         clusterId: Int? = nil,
         isMainCluster: Bool = false) {
        self.bbox = bbox
        self.confidence = confidence
        self.classId = classId
        self.className = className
        self.objConf = objConf
        self.clsConf = clsConf
        self.clusterId = clusterId
        self.isMainCluster = isMainCluster
    }

    var aspectRatio: CGFloat {
        guard bbox.height > 0 else { return 0 }
        return bbox.width / bbox.height
    }
}
