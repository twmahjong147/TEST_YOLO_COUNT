import Foundation

enum SimilarityClusterer {
    static func cluster(
        embeddings: [[Float]],
        similarityThreshold: Float = 0.80
    ) -> [Int] {
        guard embeddings.count > 1 else {
            return embeddings.isEmpty ? [] : [0]
        }
        
        let distanceThreshold = 1.0 - similarityThreshold
        
        var clusters: [[Int]] = embeddings.indices.map { [$0] }
        
        while true {
            var minDistance: Float = .infinity
            var mergePair: (Int, Int)?
            
            for i in 0..<clusters.count {
                for j in (i + 1)..<clusters.count {
                    let distance = averageLinkageDistance(
                        cluster1: clusters[i],
                        cluster2: clusters[j],
                        embeddings: embeddings
                    )
                    
                    if distance < minDistance {
                        minDistance = distance
                        mergePair = (i, j)
                    }
                }
            }
            
            guard let (i, j) = mergePair, minDistance <= distanceThreshold else {
                break
            }
            
            clusters[i].append(contentsOf: clusters[j])
            clusters.remove(at: j)
        }
        
        var labels = Array(repeating: -1, count: embeddings.count)
        
        for (clusterID, cluster) in clusters.enumerated() {
            for index in cluster {
                labels[index] = clusterID
            }
        }
        
        return labels
    }
    
    private static func averageLinkageDistance(
        cluster1: [Int],
        cluster2: [Int],
        embeddings: [[Float]]
    ) -> Float {
        var totalDistance: Float = 0
        var count = 0
        
        for i in cluster1 {
            for j in cluster2 {
                let similarity = StatisticsHelper.cosineSimilarity(embeddings[i], embeddings[j])
                let distance = 1.0 - similarity
                totalDistance += distance
                count += 1
            }
        }
        
        return count > 0 ? totalDistance / Float(count) : .infinity
    }
}
