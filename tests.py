import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import shutil
import tempfile
from veo_generator import FileManager, MediaProcessor, GenAIService, Config

class TestFileManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_unique_filename_simple(self):
        # Test basic filename creation
        name = FileManager.get_unique_filename(self.test_dir, "test", ".txt")
        expected = os.path.join(self.test_dir, "test.txt")
        self.assertEqual(name, expected)

    def test_get_unique_filename_increment(self):
        # Create a file to force increment
        base = os.path.join(self.test_dir, "test.txt")
        with open(base, 'w') as f:
            f.write("content")
        
        name = FileManager.get_unique_filename(self.test_dir, "test", ".txt")
        expected = os.path.join(self.test_dir, "test_1.txt")
        self.assertEqual(name, expected)

    def test_get_unique_filename_increment_multiple(self):
        # Create multiple files
        with open(os.path.join(self.test_dir, "test.txt"), 'w') as f: f.write("1")
        with open(os.path.join(self.test_dir, "test_1.txt"), 'w') as f: f.write("2")
        
        name = FileManager.get_unique_filename(self.test_dir, "test", ".txt")
        expected = os.path.join(self.test_dir, "test_2.txt")
        self.assertEqual(name, expected)

    def test_cleanup_files(self):
        f1 = os.path.join(self.test_dir, "t1.txt")
        f2 = os.path.join(self.test_dir, "t2.txt")
        with open(f1, 'w') as f: f.write("1")
        with open(f2, 'w') as f: f.write("2")
        
        FileManager.cleanup_files([f1, f2])
        self.assertFalse(os.path.exists(f1))
        self.assertFalse(os.path.exists(f2))

class TestMediaProcessor(unittest.TestCase):
    @patch('cv2.VideoCapture')
    @patch('cv2.imwrite')
    def test_extract_frame_success(self, mock_imwrite, mock_videocapture):
        # Setup mock behavior
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100 # 100 frames
        mock_cap.read.return_value = (True, "fake_frame_data")
        mock_videocapture.return_value = mock_cap

        result = MediaProcessor.extract_frame("dummy_video.mp4", "out.png", "last")
        
        # Verify interactions
        mock_videocapture.assert_called_with("dummy_video.mp4")
        mock_cap.set.assert_called() # Should set position
        mock_imwrite.assert_called_with("out.png", "fake_frame_data")
        mock_cap.release.assert_called()
        self.assertEqual(result, "out.png")

    @patch('cv2.VideoCapture')
    def test_extract_frame_fail_open(self, mock_videocapture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        result = MediaProcessor.extract_frame("bad.mp4", "out.png")
        self.assertIsNone(result)

    @patch('veo_generator.VideoFileClip')
    @patch('veo_generator.concatenate_videoclips')
    def test_concatenate_videos(self, mock_concat, mock_videoclip):
        mock_clip_instance = MagicMock()
        mock_videoclip.return_value = mock_clip_instance
        
        mock_final_clip = MagicMock()
        mock_concat.return_value = mock_final_clip

        result = MediaProcessor.concatenate_videos(["v1.mp4", "v2.mp4"], "final.mp4")

        self.assertEqual(result, "final.mp4")
        self.assertEqual(mock_videoclip.call_count, 2)
        mock_concat.assert_called_once()
        mock_final_clip.write_videofile.assert_called_once()
        mock_clip_instance.close.assert_called()

class TestGenAIService(unittest.TestCase):
    def setUp(self):
        # Patch the genai.Client to avoid real network calls
        self.patcher = patch('veo_generator.genai.Client')
        self.MockClient = self.patcher.start()
        self.mock_client_instance = self.MockClient.return_value
        
        # Setup default mock responses
        self.mock_operation = MagicMock()
        self.mock_operation.done = True
        self.mock_operation.response.generated_videos = [MagicMock()]
        self.mock_client_instance.models.generate_videos.return_value = self.mock_operation
        
        self.service = GenAIService()

    def tearDown(self):
        self.patcher.stop()

    def test_generate_video_simple(self):
        with patch('builtins.open', mock_open(read_data=b"data")) as m:
            # We don't provide image_path here, so open shouldn't be called for image
            result = self.service.generate_video("test prompt", "out.mp4", resolution="720p")
            
        self.mock_client_instance.models.generate_videos.assert_called_once()
        call_args = self.mock_client_instance.models.generate_videos.call_args
        self.assertEqual(call_args.kwargs['model'], Config.MODEL_VIDEO)
        self.assertEqual(call_args.kwargs['prompt'], "test prompt")
        # Check config for resolution
        self.assertEqual(call_args.kwargs['config'].resolution, "720p")
        self.assertEqual(result, "out.mp4")

    def test_generate_video_resolution_change(self):
        result = self.service.generate_video("test", "out.mp4", resolution="1080p")
        call_args = self.mock_client_instance.models.generate_videos.call_args
        self.assertEqual(call_args.kwargs['config'].resolution, "1080p")

    def test_generate_video_with_image(self):
        with patch('builtins.open', mock_open(read_data=b"fake_image_bytes")):
            result = self.service.generate_video(
                "test prompt", 
                "out.mp4", 
                image_path="test.png",
                aspect_ratio="16:9",
                resolution="720p"
            )

        call_args = self.mock_client_instance.models.generate_videos.call_args
        self.assertIn('image', call_args.kwargs)
        self.assertIsNotNone(call_args.kwargs['image'])
        self.assertEqual(result, "out.mp4")

    def test_generate_image(self):
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = True
        mock_image_obj = MagicMock()
        mock_part.as_image.return_value = mock_image_obj
        mock_response.parts = [mock_part]
        
        self.mock_client_instance.models.generate_content.return_value = mock_response

        result = self.service.generate_image("prompt", "img.png")
        
        self.mock_client_instance.models.generate_content.assert_called_once()
        mock_image_obj.save.assert_called_with("img.png")
        self.assertEqual(result, "img.png")

    def test_generate_video_api_error(self):
        self.mock_client_instance.models.generate_videos.side_effect = Exception("API Error")
        result = self.service.generate_video("prompt", "out.mp4")
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
