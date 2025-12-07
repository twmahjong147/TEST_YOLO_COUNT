### 1. Core Detection & "Main" Object Logic
*   **FR-01: Single "Main" Object Enforceability**
    *   The program must identify and report only **one** type of object per image.
*   **FR-02: The "Main" Object Algorithm**
    *   To determine which object is the "Main" one, the app must calculate a score for every detected class group:
    *   **Formula:** `Score = (Total Converge Area of Bounding Boxes) × (Object Count)`.
    *   The class with the highest score wins; all others are discarded.

### 2. Auto-Detection & Precision Mode
*   **FR-03: Default Auto-Detection**
    *   By default, the user **does not** need to provide any text prompt.
    *   The program automatically detects objects using a **Hybrid Vocabulary System**:
*   **FR-04: Manual Precision (Override)**
*   **Text Input:** Users can manually give an object name. The program will generate a text embedding on-the-fly and strictly count objects matching that description.

### 3. Output & Persistence
*   **FR-05: Result Burning**
    *   The app must generate a new image file that has the count number and bounding boxes/dots **permanently drawn (burned)** onto the original photo.

### 4. Technical Architecture (The "Free Stack")

To ensure commercial viability without licensing fees, the app will use the following pipeline:

#### 4.1. AI Pipeline
1.  **Stage 1: The Proposer (YOLOX-Nano)**
    *   *Function:* Class-Agnostic Object Detection. Finds *where* objects are.
    *   *License:* Apache 2.0.
2.  **Stage 2: The Classifier (TinyCLIP)**
    *   *Function:* Zero-Shot Classification. Compares crops against the **Hybrid Vocabulary**.
    *   *License:* MIT.
3.  **Stage 3: The Logic Layer**
    *   Applies the `Area * Count` heuristic.

