<!--
  ============================================================================
  SYNC IMPACT REPORT
  ============================================================================
  Version Change: NEW → 1.0.0
  
  Rationale: Initial constitution establishment for AICounter iOS app.
  This is the first version of the project constitution, defining core
  development principles and governance rules.
  
  Modified Principles:
  - NEW: I. Code Quality Principles
  - NEW: II. Testing Standards
  - NEW: III. User Experience Consistency
  - NEW: IV. Performance Requirements
  - NEW: V. Architecture Constraints
  
  Added Sections:
  - Core Principles (5 principles)
  - Governance
  
  Removed Sections: None (new constitution)
  
  Templates Requiring Updates:
  ✅ .specify/templates/plan-template.md - No changes needed (constitution check section is generic)
  ✅ .specify/templates/spec-template.md - No changes needed (technology-agnostic)
  ✅ .specify/templates/tasks-template.md - No changes needed (test-related guidance compatible)
  
  Follow-up TODOs: None - all placeholders filled
  ============================================================================
-->

# AICounter Project Constitution

## Core Principles

### I. Code Quality Principles

The codebase MUST maintain high standards of clarity, maintainability, and architectural integrity.

**Non-Negotiable Rules:**

- Follow clean code practices with clear, self-documenting naming conventions
- MUST maintain single responsibility principle for all classes and functions
- Functions MUST NOT exceed 50 lines of code
- Variable and function names MUST be meaningful and self-documenting
- Nesting depth MUST NOT exceed 3 levels in any code block
- Complex algorithms and business logic MUST be documented with inline comments
- Platform-specific best practices MUST be followed (Swift for iOS, Python for ML components)

**Rationale:** Clean, maintainable code reduces technical debt, enables faster iteration, and ensures the ML pipeline remains debuggable and extensible. Given the complexity of the YOLOX and TinyCLIP integration, code clarity is critical for troubleshooting and optimization.

---

### II. Testing Standards

All code changes MUST be validated through comprehensive automated testing to ensure reliability and correctness.

**Non-Negotiable Rules:**

- MUST maintain minimum 80% code coverage for critical paths (object detection, embedding extraction, clustering)
- MUST write unit tests for all business logic before implementation
- MUST include integration tests for API and model interactions (YOLOX, TinyCLIP, Core Data)
- MUST test edge cases and error conditions (empty images, invalid inputs, memory limits)
- Test names MUST be descriptive and explain what is being tested
- External dependencies MUST be mocked in unit tests
- ML model inference MUST be performance tested with target < 100ms per frame
- Test-Driven Development (TDD) is RECOMMENDED: write failing tests before implementation

**Rationale:** Given the ML inference pipeline's complexity and performance requirements, rigorous testing prevents regressions and ensures the app meets sub-2-second processing targets. Integration tests are critical for validating the multi-stage pipeline (detection → embedding → clustering).

---

### III. User Experience Consistency

The application MUST deliver a consistent, accessible, and responsive experience that adheres to platform standards.

**Non-Negotiable Rules:**

- MUST follow iOS Human Interface Guidelines in all UI implementations
- MUST maintain consistent navigation patterns throughout the app
- MUST provide immediate visual feedback for all user actions
- MUST handle loading states gracefully with progress indicators
- MUST handle error states gracefully with clear, actionable messages
- MUST support accessibility features (VoiceOver, Dynamic Type, minimum contrast ratios)
- Animations MUST run at 60fps minimum (no dropped frames during transitions)
- UI MUST be responsive and adapt to different screen sizes (iPhone, iPad)

**Rationale:** User trust depends on predictable, polished interactions. Since the app processes images with ML models, users need clear feedback during potentially multi-second operations. Accessibility compliance ensures the app serves all users.

---

### IV. Performance Requirements

The application MUST meet strict performance benchmarks to ensure responsiveness and battery efficiency.

**Non-Negotiable Rules:**

- App launch time MUST be < 2 seconds (cold start to interactive state)
- ML model inference MUST be < 100ms per frame (YOLOX + TinyCLIP total pipeline)
- Memory usage MUST be < 150MB during normal operation (with both models loaded)
- Camera and ML processing MUST be battery efficient (< 2% battery per 10 image sessions)
- CoreML models MUST be optimized for size and inference speed
- Performance bottlenecks MUST be profiled using Instruments and eliminated
- Results MUST be cached appropriately to reduce redundant processing

**Rationale:** Poor performance erodes user experience and limits device compatibility. The 100ms inference target ensures real-time feel, and memory constraints prevent crashes on older devices. These targets are validated in MODEL_TESTING_RESULTS.md.

---

### V. Architecture Constraints

The project MUST follow standard iOS architectural patterns and minimize external dependencies.

**Non-Negotiable Rules:**

- DO NOT use Swift Package Management architecture (standard Xcode project structure REQUIRED)
- MUST use standard Xcode project structure (.xcodeproj or .xcworkspace at root)
- External dependencies MUST be minimal and well-justified with documented rationale
- MUST prefer built-in frameworks (CoreML, Vision, UIKit, SwiftUI) over third-party libraries
- If third-party libraries are required, MUST use CocoaPods or manual framework integration (NOT Swift Package Manager)
- New dependencies MUST be approved via code review with justification

**Rationale:** Standard Xcode project structure ensures compatibility with Apple tooling and simplifies onboarding. Minimizing dependencies reduces security surface area, simplifies maintenance, and avoids version conflicts. Built-in frameworks are optimized for Apple Silicon and receive ongoing support.

---

## Governance

### Amendment Process

This constitution is the supreme authority for development practices and code standards. Changes to the constitution require:

1. Written proposal documenting the change rationale
2. Review and approval from project maintainers
3. Migration plan if existing code must be updated
4. Version bump following semantic versioning (see below)

### Versioning Policy

Constitution versions follow MAJOR.MINOR.PATCH:

- **MAJOR**: Backward-incompatible governance changes, principle removals, or redefinitions
- **MINOR**: New principles added, sections expanded, materially new guidance
- **PATCH**: Clarifications, wording improvements, typo fixes, non-semantic refinements

### Compliance Review

- All pull requests MUST be verified for constitution compliance during code review
- Violations MUST be justified in writing and approved by maintainers
- Complexity additions MUST be justified with documented rationale
- Testing requirements are validated via automated coverage reports

### Runtime Guidance

For day-to-day development guidance beyond constitutional principles, refer to:

- `AICounter_PRD.md` - Product requirements and feature specifications
- `SWIFT_MIGRATION_GUIDE.md` - Implementation patterns for ML integration
- `COREML_CONVERSION_SUMMARY.md` - Model architecture and conversion details
- `README_COREML.md` - Quick start and usage instructions

---

**Version**: 1.0.0 | **Ratified**: 2025-12-09 | **Last Amended**: 2025-12-09
