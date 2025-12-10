import UIKit
import CoreGraphics
import CoreImage
import VideoToolbox
import Accelerate
import Photos

enum ImageProcessor {
    static func createPixelBuffer(from image: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }

    static func resizeImage(_ image: CGImage, to size: CGSize) -> CGImage? {
        let width = Int(size.width)
        let height = Int(size.height)
        
        guard let colorSpace = image.colorSpace else { return nil }
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: image.bitsPerComponent,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: image.bitmapInfo.rawValue
        ) else {
            return nil
        }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return context.makeImage()
    }

    static func preProc(_ image: CGImage, to size: CGSize) -> ([Float], CGFloat)? {
        let targetWidth = Int(size.width)
        let targetHeight = Int(size.height)
        let originalWidth = image.width
        let originalHeight = image.height

        // Calculate scale ratio
        let scale = min(CGFloat(targetWidth) / CGFloat(originalWidth), CGFloat(targetHeight) / CGFloat(originalHeight))
        let resizedWidth = Int(CGFloat(originalWidth) * scale)
        let resizedHeight = Int(CGFloat(originalHeight) * scale)

        // Create padded background with constant value 114 (RGB)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            return nil
        }

        // Fill with 114 gray background using CGContext APIs to avoid byte-order issues
        let gray = CGFloat(114.0 / 255.0)
        context.setFillColor(CGColor(red: gray, green: gray, blue: gray, alpha: 1.0))
        context.fill(CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

        // Draw resized image centered on padded background
        // Place at top-left (0,0) to match Python
        let xOffset = 0
        let yOffset = 0
        context.interpolationQuality = .high
        context.draw(
            image,
            in: CGRect(x: xOffset, y: yOffset, width: resizedWidth, height: resizedHeight)
        )

        guard let paddedImage = context.makeImage() else { return nil }

        // Save paddedImage to Photos for debugging (requires Info.plist NSPhotoLibraryAddUsageDescription)
        // let uiImage = UIImage(cgImage: paddedImage)
        // PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
        //     switch status {
        //     case .authorized, .limited:
        //         PHPhotoLibrary.shared().performChanges({
        //             PHAssetChangeRequest.creationRequestForAsset(from: uiImage)
        //         }) { success, error in
        //             if let error = error {
        //                 print("Failed to save paddedImage to Photos: \(error)")
        //             }
        //         }
        //     default:
        //         print("Photo library access not granted: \(status.rawValue)")
        //     }
        // }

        // Convert to [Float] in CHW order
        let width = targetWidth
        let height = targetHeight
        var chwBuffer = [Float](repeating: 0, count: 3 * width * height)
        guard let dataProvider = paddedImage.dataProvider,
              let data = dataProvider.data else { return nil }
        // Ensure non-nil byte pointer
        guard let ptr = CFDataGetBytePtr(data) else { return nil }

        // Inspect bitmap info to determine byte order and channel layout
        let imgBitmapInfo = paddedImage.bitmapInfo
        let alphaInfoRaw = imgBitmapInfo.rawValue & CGBitmapInfo.alphaInfoMask.rawValue
        let alphaInfo = CGImageAlphaInfo(rawValue: alphaInfoRaw) ?? .none
        let isLittleEndian = imgBitmapInfo.contains(.byteOrder32Little)

        // Determine channel indices for R,G,B within the 4-byte pixel tuple
        var rIndex = 2
        var gIndex = 1
        var bIndex = 0

        if isLittleEndian {
            // Common case on iOS/macOS: little-endian + alpha-first/noneSkipFirst -> BGRA (B=0,G=1,R=2,A=3)
            if alphaInfo == .premultipliedFirst || alphaInfo == .noneSkipFirst || alphaInfo == .first {
                bIndex = 0; gIndex = 1; rIndex = 2
            } else {
                // little-endian + alpha-last or premultipliedLast -> RGBA (R=0,G=1,B=2,A=3)
                rIndex = 0; gIndex = 1; bIndex = 2
            }
        } else {
            // Big-endian layouts are less common; handle typical ARGB/ABGR variants conservatively
            if alphaInfo == .premultipliedFirst || alphaInfo == .first {
                // ARGB -> A=0,R=1,G=2,B=3
                rIndex = 1; gIndex = 2; bIndex = 3
            } else {
                // Fallback assume RGBA order
                rIndex = 0; gIndex = 1; bIndex = 2
            }
        }

        for y in 0..<height {
            for x in 0..<width {
                let pixel = ((y * width) + x) * 4
                let r = Float(ptr[pixel + rIndex])
                let g = Float(ptr[pixel + gIndex])
                let b = Float(ptr[pixel + bIndex])
                // CHW: [C][H][W] in BGR order for YOLOX
                chwBuffer[0 * width * height + y * width + x] = b
                chwBuffer[1 * width * height + y * width + x] = g
                chwBuffer[2 * width * height + y * width + x] = r
            }
        }
        // Convert to float32
        return (chwBuffer, scale)
    }
    
    static func cropImage(_ image: CGImage, to rect: CGRect) -> CGImage? {
        let adjustedRect = CGRect(
            x: max(0, rect.origin.x),
            y: max(0, rect.origin.y),
            width: min(rect.width, CGFloat(image.width) - rect.origin.x),
            height: min(rect.height, CGFloat(image.height) - rect.origin.y)
        )
        
        return image.cropping(to: adjustedRect)
    }
    
    static func normalizeEmbedding(_ vector: [Float]) -> [Float] {
        var result = vector
        var magnitude: Float = 0
        
        vDSP_svesq(vector, 1, &magnitude, vDSP_Length(vector.count))
        magnitude = sqrt(magnitude)
        
        guard magnitude > 0 else { return vector }
        
        var inverseMag = 1.0 / magnitude
        vDSP_vsmul(vector, 1, &inverseMag, &result, 1, vDSP_Length(vector.count))
        
        return result
    }
}
