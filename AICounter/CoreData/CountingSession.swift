import Foundation
import CoreData

@objc(CountingSession)
class CountingSession: NSManagedObject {
    @NSManaged var id: UUID
    @NSManaged var thumbnailData: Data
    @NSManaged var count: Int16
    @NSManaged var timestamp: Date
    @NSManaged var confidenceThreshold: Float
    @NSManaged var similarityThreshold: Float
}

extension CountingSession: Identifiable {}
