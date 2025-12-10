import SwiftUI
import PhotosUI
import Photos

struct ContentView: View {
    @State private var selectedImage: UIImage?
    @State private var countResult: CountResult?
    @State private var isProcessing = false
    @State private var showCamera = false
    @State private var showHistory = false
    @State private var showPhotoPicker = false
    @State private var errorMessage: String?
    @State private var showError = false
    
    @State private var aiCounter = AICounter()
    @State private var historyManager = HistoryManager()
    
    private let confidenceThreshold: Float = 0.001
    private let similarityThreshold: Float = 0.80
    
    init() {}
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    if let image = selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 300)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .shadow(radius: 4)
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.gray.opacity(0.2))
                            .frame(height: 250)
                            .overlay {
                                VStack(spacing: 8) {
                                    Image(systemName: "photo.on.rectangle.angled")
                                        .font(.system(size: 50))
                                        .foregroundStyle(.secondary)
                                    Text("No Image Selected")
                                        .foregroundStyle(.secondary)
                                }
                            }
                    }
                    
                    if let result = countResult {
                        VStack(spacing: 12) {
                            Text("\(result.count)")
                                .font(.system(size: 72, weight: .bold))
                                .foregroundStyle(.primary)
                            
                            Text("Objects Counted")
                                .font(.title3)
                                .foregroundStyle(.secondary)
                            
                            Text("Processing time: \(String(format: "%.2f", result.processingTime))s")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    
                    if isProcessing {
                        ProgressView("Processing...")
                            .padding()
                    }
                    
                    VStack(spacing: 16) {
                        Button {
                            showCamera = true
                        } label: {
                            Label("Capture Photo", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundStyle(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        
                        Button {
                            showPhotoPicker = true
                        } label: {
                            Label("Select from Library", systemImage: "photo.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .foregroundStyle(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                        
                        if selectedImage != nil && !isProcessing {
                            Button {
                                Task {
                                    await processImage()
                                }
                            } label: {
                                Label("Count Objects", systemImage: "number.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.green)
                                    .foregroundStyle(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("AICounter")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showHistory = true
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                    }
                }
            }
            .sheet(isPresented: $showCamera) {
                CameraView { image in
                    selectedImage = image
                    countResult = nil
                }
            }
            .sheet(isPresented: $showPhotoPicker) {
                PhotoPicker(selectedImage: $selectedImage)
                    .onDisappear {
                        countResult = nil
                    }
            }
            .sheet(isPresented: $showHistory) {
                HistoryView()
            }
            .alert("Error", isPresented: $showError) {
                Button("OK") {}
            } message: {
                if let errorMessage = errorMessage {
                    Text(errorMessage)
                }
            }
        }
        .task {
            do {
                try await aiCounter.loadModels()
            } catch {
                errorMessage = error.localizedDescription
                showError = true
            }
        }
    }
    
    // Render visualization image from CountResult
    private func renderVisualization(source: UIImage, result: CountResult) -> UIImage {
        let size = source.size
        UIGraphicsBeginImageContextWithOptions(size, false, source.scale)
        source.draw(at: .zero)
        guard let context = UIGraphicsGetCurrentContext() else {
            return source
        }

        func colorForCluster(_ id: Int) -> UIColor {
            let hue = CGFloat((abs(id * 57) % 360)) / 360.0
            return UIColor(hue: hue, saturation: 0.7, brightness: 0.9, alpha: 0.35)
        }

        for det in result.detections {
            let rect = det.bbox
            let clusterId = det.clusterId ?? -1
            let isMain = det.isMainCluster
            let fillColor = colorForCluster(clusterId)

            context.setFillColor(fillColor.cgColor)
            context.fill(rect)

            context.setStrokeColor((isMain ? UIColor.red : UIColor.white).cgColor)
            context.setLineWidth(isMain ? 3.0 : 1.5)
            context.stroke(rect)

            let label = det.className ?? "cls_\(det.classId)"
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 12),
                .foregroundColor: UIColor.white
            ]
            let textSize = (label as NSString).size(withAttributes: attrs)
            let textRect = CGRect(x: rect.origin.x + 4, y: rect.origin.y + 4, width: textSize.width + 8, height: textSize.height + 4)
            context.setFillColor(UIColor.black.withAlphaComponent(0.6).cgColor)
            context.fill(textRect)
            (label as NSString).draw(in: CGRect(x: textRect.origin.x + 4, y: textRect.origin.y + 2, width: textSize.width, height: textSize.height), withAttributes: attrs)
        }

        let annotated = UIGraphicsGetImageFromCurrentImageContext() ?? source
        UIGraphicsEndImageContext()
        return annotated
    }

    private func processImage() async {
        guard let image = selectedImage?.cgImage else { return }
        
        isProcessing = true
        defer { isProcessing = false }
        
        do {
            let result = try await aiCounter.count(
                image: image,
                confidenceThreshold: confidenceThreshold,
                similarityThreshold: similarityThreshold
            )
            
            countResult = result
            
            historyManager.saveSession(
                result: result,
                image: image,
                confidenceThreshold: confidenceThreshold,
                similarityThreshold: similarityThreshold
            )
            
            // Generate visualization with cluster highlighting (mirror Python visualize_detections_custom)
            if let ui = selectedImage {
                Task { @MainActor in
                    let vis = renderVisualization(source: ui, result: result)
                    // Save to Photo Library
                    PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                        if status == .authorized || status == .limited {
                            PHPhotoLibrary.shared().performChanges({
                                PHAssetChangeRequest.creationRequestForAsset(from: vis)
                            }, completionHandler: { success, error in
                                if let error = error {
                                    print("Failed saving visualization to Photos: \(error)")
                                }
                            })
                        } else {
                            print("Photo library access not granted: \(status.rawValue)")
                        }
                    }
                    // Show visualization in UI
                    selectedImage = vis
                }
            }
        } catch {
            errorMessage = error.localizedDescription
            showError = true
        }
    }
}

struct PhotoPicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) private var dismiss
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PhotoPicker
        
        init(_ parent: PhotoPicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.dismiss()
            
            guard let provider = results.first?.itemProvider else { return }
            
            if provider.canLoadObject(ofClass: UIImage.self) {
                provider.loadObject(ofClass: UIImage.self) { obj, _ in
                    guard let image = obj as? UIImage else { return }
                    Task { @MainActor in
                        self.parent.selectedImage = image
                    }
                }
            }
        }
    }
}
