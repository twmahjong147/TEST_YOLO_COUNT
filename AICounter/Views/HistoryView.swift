import SwiftUI

struct HistoryView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var sessions: [CountingSession] = []
    @State private var showClearAlert = false
    
    private let historyManager = HistoryManager()
    
    init() {}
    
    var body: some View {
        NavigationStack {
            Group {
                if sessions.isEmpty {
                    ContentUnavailableView(
                        "No History",
                        systemImage: "clock.arrow.circlepath",
                        description: Text("Your counting history will appear here")
                    )
                } else {
                    List {
                        ForEach(sessions) { session in
                            HistoryRowView(session: session)
                        }
                        .onDelete(perform: deleteSessions)
                    }
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .topBarTrailing) {
                    if !sessions.isEmpty {
                        Button("Clear All", role: .destructive) {
                            showClearAlert = true
                        }
                    }
                }
            }
            .alert("Clear All History?", isPresented: $showClearAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Clear All", role: .destructive) {
                    historyManager.clearAll()
                    loadHistory()
                }
            } message: {
                Text("This will delete all \(sessions.count) counting sessions. This action cannot be undone.")
            }
        }
        .task {
            loadHistory()
        }
    }
    
    private func loadHistory() {
        sessions = historyManager.fetchHistory()
    }
    
    private func deleteSessions(at offsets: IndexSet) {
        for index in offsets {
            let session = sessions[index]
            historyManager.deleteSession(session)
        }
        loadHistory()
    }
}

struct HistoryRowView: View {
    let session: CountingSession
    
    var body: some View {
        HStack(spacing: 12) {
            if let image = UIImage(data: session.thumbnailData) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 60, height: 60)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 60, height: 60)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text("Count: \(session.count)")
                    .font(.headline)
                
                Text(session.timestamp, style: .date)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                Text(session.timestamp, style: .time)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            Spacer()
        }
        .padding(.vertical, 4)
    }
}
