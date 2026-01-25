# Veo3 Pro Generator

Advanced AI Media Generation tool powered by Google Gemini (Veo 3.1, Imagen 4, Lyria). This tool allows you to generate videos, images, and audio, link them into chains, create smooth transitions (bridges), and manage your media workflow efficiently.

## 🚀 Installation & Setup

1.  **Prerequisites**: Python 3.10+ installed.
2.  **Clone the repository**:
    ```bash
    git clone https://github.com/kubiknubika/Veo3.git
    cd Veo3
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Setup**:
    Create a `.env` file in the root directory and add your API key:
    ```env
    GEMINI_API_KEY=your_google_api_key_here
    ```

## ▶️ Usage

Run the main application using Python:

```powershell
python veo_generator.py
```

*Note: You may want to create a `run.bat` or `run.sh` script locally to execute this with your specific Python environment path if needed.*

## ✨ Features

*   **Text-to-Video (Veo 3.1)**: Generate high-quality videos with resolution control (720p/1080p).
*   **Image Generation (Imagen 4 / NanoBanana)**: Create start/end frames or standalone art.
*   **Audio Generation (Lyria)**: Experimental model, currently generates low quality audio. For high quality audio, please use third-party models.
*   **Video Chaining**: Create long sequences where the last frame of one video becomes the start of the next.
*   **Bridge & Interpolation**: Smoothly connect two video clips (existing or generated) with a generated transition. Includes a **Verification Loop** to perfect the result.
*   **Batch Processing**: Generate multiple videos from a list of prompts or concatenate existing files.

## 🆕 What's Changed (v2 vs v1)

Major refactoring and feature expansion compared to the initial version:

### 1. 🏗️ Architecture & Organization
*   **Refactored Codebase**: Moved from a simple script to a modular Object-Oriented design (`GenAIService`, `MediaProcessor`, `FileManager`).
*   **Smart File Management**: All outputs are now organized into `output/videos`, `output/images`, and `output/audio`.
*   **Conflict Resolution**: Automatic file renaming (incrementing counters) and safe cleanup of temporary files.

### 2. 🎬 Advanced Video Control
*   **Resolution Selection**: Choose between **720p** (quota-friendly) and **1080p** (HD).
*   **Negative Prompts**: Added support for negative prompts.
    *   *Auto-Generate*: If left empty, the system uses **Gemini 2.0 Flash** to analyze your prompt and generate a relevant negative prompt automatically.
*   **Constraints**:
    *   **Last Frame**: Force a specific ending image (from file or generated on the fly).
    *   **Reference Images**: Upload up to 3 images to guide style and content.
*   **Safety**: Enabled `person_generation="ALLOW_ADULT"` by default.
*   **Validation**: Added automatic token counting to warn if prompts exceed API limits.

### 3. 🌉 New Bridge Mode (Interpolation)
*   **Old v1**: Simple start-to-end generation.
*   **New v2**: **Verification Loop**.
    *   Generates a bridge between V1 and V2.
    *   Stitches a preview.
    *   Asks: *"Are you satisfied?"*
    *   If **No**: You can keep the bridge but **regenerate the second video** (using the bridge's end frame) infinitely until the transition is perfect.

### 4. 🧪 Testing
*   Added `tests.py`: Comprehensive mocked unit tests for safe logic verification.
*   Added `api_tests.py`: Real API tests to verify connectivity and quota usage.
