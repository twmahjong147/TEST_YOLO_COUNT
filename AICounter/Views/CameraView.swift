import SwiftUI
import AVFoundation

struct CameraView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var cameraManager = CameraManager()
    @State private var session: AVCaptureSession?
    @State private var showError = false
    @State private var errorMessage = ""
    
    let onImageCaptured: (UIImage) -> Void
    
    init(onImageCaptured: @escaping (UIImage) -> Void) {
        self.onImageCaptured = onImageCaptured
    }
    
    var body: some View {
        ZStack {
            if let session = session {
                CameraPreviewView(session: session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
            }
            
            VStack {
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.title2)
                            .foregroundStyle(.white)
                            .padding()
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                    .padding()
                    
                    Spacer()
                }
                
                Spacer()
                
                Button {
                    Task {
                        if let image = await cameraManager.capturePhoto() {
                            cameraManager.stopSession()
                            onImageCaptured(image)
                            dismiss()
                        }
                    }
                } label: {
                    Circle()
                        .fill(Color.white)
                        .frame(width: 70, height: 70)
                        .overlay {
                            Circle()
                                .stroke(Color.white, lineWidth: 3)
                                .frame(width: 80, height: 80)
                        }
                }
                .padding(.bottom, 40)
            }
        }
        .task {
            await cameraManager.checkAuthorization()
            
            if cameraManager.isAuthorized {
                do {
                    let captureSession = try await cameraManager.setupSession()
                    session = captureSession
                    captureSession.startRunning()
                } catch {
                    errorMessage = error.localizedDescription
                    showError = true
                }
            } else {
                errorMessage = "Camera access denied"
                showError = true
            }
        }
        .onDisappear {
            cameraManager.stopSession()
        }
        .alert("Camera Error", isPresented: $showError) {
            Button("OK") {
                dismiss()
            }
        } message: {
            Text(errorMessage)
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        context.coordinator.previewLayer = previewLayer
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            context.coordinator.previewLayer?.frame = uiView.bounds
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}
