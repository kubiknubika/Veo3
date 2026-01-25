import unittest
import os
import time
from veo_generator import GenAIService, Config, FileManager

class TestRealAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure API key is present
        if not os.environ.get("GEMINI_API_KEY"):
            print("WARNING: GEMINI_API_KEY not found. Skipping real API tests.")
            raise unittest.SkipTest("GEMINI_API_KEY not found in environment")
        
        cls.service = GenAIService()
        Config.ensure_dirs()

    def test_real_image_generation(self):
        print("\n--- Testing Real Image Generation ---")
        prompt = "A futuristic city with flying cars, pixel art style"
        filename = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "api_test_image", ".png")
        
        result = self.service.generate_image(prompt, filename)
        
        self.assertIsNotNone(result, "Image generation returned None")
        self.assertTrue(os.path.exists(result), "Generated image file does not exist")
        print(f"Image generated successfully at {result}")

    def test_real_audio_generation(self):
        print("\n--- Testing Real Audio Generation ---")
        prompt = "Electronic beat, fast tempo, 120 bpm"
        filename = FileManager.get_unique_filename(Config.OUTPUT_DIRS["audio"], "api_test_audio", ".wav")
        
        # 5 seconds duration to save time/quota
        result = self.service.generate_audio(prompt, duration_seconds=5, output_filename=filename)
        
        self.assertIsNotNone(result, "Audio generation returned None")
        self.assertTrue(os.path.exists(result), "Generated audio file does not exist")
        print(f"Audio generated successfully at {result}")

    def test_real_video_generation(self):
        print("\n--- Testing Real Video Generation (This consumes significant quota and time) ---")
        prompt = "A drop of water falling into a puddle, slow motion, 4k"
        filename = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "api_test_video", ".mp4")
        
        # Use 720p for testing to save quota/resources if supported
        result = self.service.generate_video(prompt, filename, resolution="720p")
        
        self.assertIsNotNone(result, "Video generation returned None")
        self.assertTrue(os.path.exists(result), "Generated video file does not exist")
        print(f"Video generated successfully at {result}")

if __name__ == "__main__":
    unittest.main()
