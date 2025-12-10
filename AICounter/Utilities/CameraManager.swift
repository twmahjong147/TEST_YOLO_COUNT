import AVFoundation
import UIKit

@MainActor
final class CameraManager: NSObject, ObservableObject {
    @Published var capturedImage: UIImage?
    @Published var isAuthorized = false
    
    private var captureSession: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var continuation: CheckedContinuation<UIImage?, Never>?
    
    override init() {
        super.init()
    }
    
    func checkAuthorization() async {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
        case .notDetermined:
            isAuthorized = await AVCaptureDevice.requestAccess(for: .video)
        default:
            isAuthorized = false
        }
    }
    
    func setupSession() async throws -> AVCaptureSession {
        let session = AVCaptureSession()
        session.beginConfiguration()
        
        session.sessionPreset = .photo
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw NSError(domain: "CameraManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "No camera available"])
        }
        
        let videoInput = try AVCaptureDeviceInput(device: videoDevice)
        
        guard session.canAddInput(videoInput) else {
            throw NSError(domain: "CameraManager", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not add video input"])
        }
        
        session.addInput(videoInput)
        
        let output = AVCapturePhotoOutput()
        
        guard session.canAddOutput(output) else {
            throw NSError(domain: "CameraManager", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not add photo output"])
        }
        
        session.addOutput(output)
        
        session.commitConfiguration()
        
        self.captureSession = session
        self.photoOutput = output
        
        return session
    }
    
    func capturePhoto() async -> UIImage? {
        guard let photoOutput = photoOutput else { return nil }
        
        let settings = AVCapturePhotoSettings()
        settings.flashMode = .auto
        
        return await withCheckedContinuation { continuation in
            self.continuation = continuation
            photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }
    
    func stopSession() {
        captureSession?.stopRunning()
        captureSession = nil
        photoOutput = nil
    }
}

extension CameraManager: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        let imageData = photo.fileDataRepresentation()
        
        Task { @MainActor in
            guard error == nil,
                  let data = imageData,
                  let image = UIImage(data: data) else {
                continuation?.resume(returning: nil)
                continuation = nil
                return
            }
            
            continuation?.resume(returning: image)
            continuation = nil
        }
    }
}
