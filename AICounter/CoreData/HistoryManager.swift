import CoreData
import UIKit

@MainActor
final class HistoryManager {
    private let persistenceController = PersistenceController.shared
    private let maxHistoryCount = 100
    
    func saveSession(
        result: CountResult,
        image: CGImage,
        confidenceThreshold: Float,
        similarityThreshold: Float
    ) {
        let context = persistenceController.viewContext
        
        let session = CountingSession(context: context)
        session.id = UUID()
        session.count = Int16(result.count)
        session.timestamp = Date()
        session.confidenceThreshold = confidenceThreshold
        session.similarityThreshold = similarityThreshold
        
        if let thumbnailData = generateThumbnail(
            from: image,
            detections: result.detections,
            clusterLabels: result.clusterLabels,
            largestClusterId: result.largestClusterId
        ) {
            session.thumbnailData = thumbnailData
        }
        
        persistenceController.save()
        
        Task {
            await enforceHistoryLimit()
        }
    }
    
    func fetchHistory(limit: Int = 100) -> [CountingSession] {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<CountingSession>(entityName: "CountingSession")
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        request.fetchLimit = limit
        
        do {
            return try context.fetch(request)
        } catch {
            print("Failed to fetch history: \(error)")
            return []
        }
    }
    
    func deleteSession(_ session: CountingSession) {
        let context = persistenceController.viewContext
        context.delete(session)
        persistenceController.save()
    }
    
    func clearAll() {
        let context = persistenceController.viewContext
        let fetchRequest = NSFetchRequest<NSFetchRequestResult>(entityName: "CountingSession")
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
        
        do {
            try context.execute(deleteRequest)
            persistenceController.save()
        } catch {
            print("Failed to clear history: \(error)")
        }
    }
    
    private func enforceHistoryLimit() async {
        let context = persistenceController.viewContext
        let request = NSFetchRequest<CountingSession>(entityName: "CountingSession")
        request.sortDescriptors = [NSSortDescriptor(key: "timestamp", ascending: false)]
        
        do {
            let allSessions = try context.fetch(request)
            
            if allSessions.count > maxHistoryCount {
                let sessionsToDelete = allSessions.dropFirst(maxHistoryCount)
                
                for session in sessionsToDelete {
                    context.delete(session)
                }
                
                persistenceController.save()
            }
        } catch {
            print("Failed to enforce history limit: \(error)")
        }
    }
    
    private func generateThumbnail(
        from image: CGImage,
        detections: [Detection],
        clusterLabels: [Int],
        largestClusterId: Int
    ) -> Data? {
        let mainDetections = zip(detections, clusterLabels).filter { $0.1 == largestClusterId }
        
        guard let firstMain = mainDetections.first else { return nil }
        
        guard let cropped = ImageProcessor.cropImage(image, to: firstMain.0.bbox) else {
            return nil
        }
        
        let targetSize = CGSize(width: 200, height: 200)
        let scale = min(targetSize.width / CGFloat(cropped.width), targetSize.height / CGFloat(cropped.height))
        
        let newWidth = Int(CGFloat(cropped.width) * scale)
        let newHeight = Int(CGFloat(cropped.height) * scale)
        
        guard let resized = ImageProcessor.resizeImage(cropped, to: CGSize(width: newWidth, height: newHeight)) else {
            return nil
        }
        
        let xOffset = (200 - newWidth) / 2
        let yOffset = (200 - newHeight) / 2
        
        guard let colorSpace = resized.colorSpace else { return nil }
        
        guard let context = CGContext(
            data: nil,
            width: 200,
            height: 200,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: 200, height: 200))
        
        context.draw(resized, in: CGRect(x: xOffset, y: yOffset, width: newWidth, height: newHeight))
        
        guard let thumbnail = context.makeImage() else { return nil }
        
        let uiImage = UIImage(cgImage: thumbnail)
        return uiImage.jpegData(compressionQuality: 0.8)
    }
}
