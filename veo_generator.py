import os
import sys
import time
import cv2
import asyncio
import wave
import shutil
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
from moviepy import VideoFileClip, concatenate_videoclips

# Load environment variables
load_dotenv()

# --- Configuration Constants ---
class Config:
    MODEL_VIDEO = "veo-3.1-generate-preview"
    MODEL_IMAGE = "gemini-2.5-flash-image"
    MODEL_TEXT = "gemini-3.0-flash-preview"
    MODEL_AUDIO = "models/lyria-realtime-exp"
    TOKEN_LIMIT = 1000
    
    if getattr(sys, 'frozen', False):
        # Running as compiled exe: use directory of the executable
        BASE_DIR = os.path.dirname(sys.executable)
    else:
        # Running as script: use directory of the script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    OUTPUT_DIRS = {
        "video": os.path.join(BASE_DIR, "output", "videos"),
        "image": os.path.join(BASE_DIR, "output", "images"),
        "audio": os.path.join(BASE_DIR, "output", "audio"),
    }
    
    @classmethod
    def ensure_dirs(cls):
        for path in cls.OUTPUT_DIRS.values():
            os.makedirs(path, exist_ok=True)

# --- File Utilities ---
class FileManager:
    @staticmethod
    def get_unique_filename(folder, base_name, extension, strict=False):
        """
        Generates a unique filename in the given folder.
        If 'strict' is True, prompts the user to enter a new name if collision occurs.
        Otherwise, appends an incrementing counter (e.g., video_1.mp4).
        """
        if not base_name:
            base_name = "generated"
        
        # Clean extension
        if not extension.startswith("."):
            extension = "." + extension

        # Remove extension from base_name if user typed it
        if base_name.lower().endswith(extension):
            base_name = base_name[:-len(extension)]

        full_path = os.path.join(folder, base_name + extension)

        if strict:
            while os.path.exists(full_path):
                print(f"Ошибка: Файл '{base_name}{extension}' уже существует в {folder}.")
                base_name = input("Введите новое имя (без расширения): ").strip()
                full_path = os.path.join(folder, base_name + extension)
            return full_path
        
        # Auto-increment logic
        counter = 1
        new_base_name = base_name
        while os.path.exists(full_path):
            new_base_name = f"{base_name}_{counter}"
            full_path = os.path.join(folder, new_base_name + extension)
            counter += 1
        
        return full_path

    @staticmethod
    def ask_for_filename(prompt_text, default_name, folder, extension):
        name = input(prompt_text)
        if not name.strip():
            name = default_name
        return FileManager.get_unique_filename(folder, name, extension, strict=True)

    @staticmethod
    def cleanup_files(file_list):
        print("Очистка временных файлов...")
        for f in file_list:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Не удалось удалить {f}: {e}")

# --- Media Processing (OpenCV / MoviePy) ---
class MediaProcessor:
    @staticmethod
    def extract_frame(video_path, output_image_path, pos="last"):
        """Extracts first or last frame from video."""
        print(f"Извлечение {pos} кадра из {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видео {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if pos == "last":
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        else: # first
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_image_path, frame)
            cap.release()
            return output_image_path
        else:
            print("Ошибка при чтении кадра.")
            cap.release()
            return None

    @staticmethod
    def concatenate_videos(video_paths, output_filename):
        """Concatenates video files."""
        print(f"Склейка {len(video_paths)} видео...")
        try:
            clips = [VideoFileClip(path) for path in video_paths]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")
            
            # Close clips to release file handles
            for clip in clips:
                clip.close()
                
            print(f"Объединенное видео сохранено: {output_filename}")
            return output_filename
        except Exception as e:
            print(f"Ошибка при склейке видео: {e}")
            return None

# --- Generative AI Service ---
class GenAIService:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def _get_image_param(self, image_path):
        """Reads image file and returns types.Image object."""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            mime_type = "image/png"
            if image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            return types.Image(imageBytes=image_bytes, mimeType=mime_type)
        except Exception as e:
            print(f"Ошибка чтения изображения {image_path}: {e}")
            return None

    def count_tokens(self, text):
        try:
            resp = self.client.models.count_tokens(model=Config.MODEL_TEXT, contents=text)
            return resp.total_tokens
        except:
            return 0

    def generate_negative_prompt(self, video_prompt):
        try:
            prompt = f"Generate a concise comma-separated list of negative prompt keywords for a high quality video generation based on this positive prompt: '{video_prompt}'. Focus on avoiding bad quality, deformations, blurs, watermark, text, ugly, distorted. Return ONLY the keywords."
            resp = self.client.models.generate_content(model=Config.MODEL_TEXT, contents=prompt)
            return resp.text.strip()
        except:
            return None

    def generate_video(self, prompt, output_filename, image_path=None, last_frame_path=None, aspect_ratio="16:9", resolution="720p", negative_prompt=None, reference_image_paths=None):
        print(f"Генерация видео: '{prompt}' [{resolution}, {aspect_ratio}]")
        
        # Token Check
        tokens = self.count_tokens(prompt)
        if tokens > Config.TOKEN_LIMIT:
             print(f"Внимание: Промпт слишком длинный ({tokens} токенов). Лимит: {Config.TOKEN_LIMIT}.")
             # We proceed anyway but warn user, or could return None
        
        try:
            config_params = {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "person_generation": "ALLOW_ADULT", # Default allowed
            }
            
            if negative_prompt:
                config_params["negative_prompt"] = negative_prompt
            
            if reference_image_paths:
                refs = []
                for p in reference_image_paths[:3]: # Limit to 3
                    img_obj = self._get_image_param(p)
                    if img_obj:
                        refs.append(types.VideoGenerationReferenceImage(image=img_obj, reference_type="ASSET"))
                if refs:
                    config_params["reference_images"] = refs

            image_param = None
            if image_path:
                image_param = self._get_image_param(image_path)
                if not image_param: return None

            if last_frame_path:
                last_frame_param = self._get_image_param(last_frame_path)
                if last_frame_param:
                    config_params["last_frame"] = last_frame_param

            config = types.GenerateVideosConfig(**config_params)

            # Try to generate with person_generation first
            try:
                operation = self.client.models.generate_videos(
                    model=Config.MODEL_VIDEO,
                    prompt=prompt,
                    image=image_param,
                    config=config
                )
            except Exception as e:
                err_str = str(e)
                if "INVALID_ARGUMENT" in err_str and ("personGeneration" in err_str or "allow_adult" in err_str):
                    print("Ошибка аргумента person_generation. Повторяем запрос без него...")
                    # Remove person_generation and retry
                    if "person_generation" in config_params:
                        del config_params["person_generation"]
                    
                    config = types.GenerateVideosConfig(**config_params)
                    operation = self.client.models.generate_videos(
                        model=Config.MODEL_VIDEO,
                        prompt=prompt,
                        image=image_param,
                        config=config
                    )
                else:
                    raise e

            print("Ожидание генерации...")
            while not operation.done:
                time.sleep(5)
                operation = self.client.operations.get(operation)

            generated_video = operation.response.generated_videos[0]
            self.client.files.download(file=generated_video.video)
            generated_video.video.save(output_filename)
            print(f"Видео сохранено: {output_filename}")
            return output_filename
        except Exception as e:
            print(f"Ошибка генерации видео: {e}")
            return None

    def generate_image(self, prompt, output_filename):
        print(f"Генерация изображения: '{prompt}'")
        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_IMAGE,
                contents=[prompt],
                config={"response_modalities":['IMAGE']}
            )
            for part in response.parts:
                if part.inline_data is not None:
                    image = part.as_image()
                    image.save(output_filename)
                    print(f"Изображение сохранено: {output_filename}")
                    return output_filename
        except Exception as e:
            print(f"Ошибка генерации изображения: {e}")
            return None

    def generate_audio(self, prompt, duration_seconds=10, output_filename=None):
        if not output_filename:
            output_filename = FileManager.get_unique_filename(Config.OUTPUT_DIRS["audio"], "generated_music", ".wav")
            
        print(f"Генерация музыки: '{prompt}' ({duration_seconds} сек)...")
        
        async def run_audio_session():
            client_lyria = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY"),
                http_options={'api_version': 'v1alpha'}
            )
            audio_buffer = bytearray()
            try:
                async with client_lyria.aio.live.music.connect(model=Config.MODEL_AUDIO) as session:
                    await session.set_weighted_prompts(prompts=[types.WeightedPrompt(text=prompt, weight=1.0)])
                    await session.set_music_generation_config(config=types.LiveMusicGenerationConfig(bpm=120, temperature=1.0))
                    await session.play()
                    print("Запись аудиопотока...")
                    start_time = time.time()
                    async for message in session.receive():
                        if message.server_content and message.server_content.audio_chunks:
                            for chunk in message.server_content.audio_chunks:
                                audio_buffer.extend(chunk.data)
                        if time.time() - start_time > duration_seconds:
                            break
                    print("Генерация аудио завершена.")
                    return audio_buffer
            except Exception as e:
                print(f"Ошибка в сессии Lyria: {e}")
                return None

        try:
            audio_data = asyncio.run(run_audio_session())
            if audio_data:
                with wave.open(output_filename, 'wb') as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(audio_data)
                print(f"Аудио сохранено: {output_filename}")
                return output_filename
            else:
                return None
        except Exception as e:
            print(f"Ошибка при запуске генерации аудио: {e}")
            return None

# --- Main Application Controller ---
class VeoGeneratorApp:
    def __init__(self):
        Config.ensure_dirs()
        self.ai = GenAIService()
        self.running = True

    def run(self):
        while self.running:
            print("\n=== Veo3 Pro Generator (Refactored) ===")
            print("1. Одиночное видео (Text-to-Video)")
            print("2. Цепочка видео (Склейка)")
            print("3. Генерация изображения (Text-to-Image)")
            print("4. Батч режим")
            print("5. Генерация аудио (Lyria)")
            print("6. Мост между видео (Interpolation with Verification)")
            print("0. Выход")
            
            choice = input("Выберите режим: ")

            if choice == "0": self.running = False
            elif choice == "1": self.handle_single_video()
            elif choice == "2": self.handle_chain_mode()
            elif choice == "3": self.handle_image_gen()
            elif choice == "4": self.handle_batch_mode()
            elif choice == "5": self.handle_audio_gen()
            elif choice == "6": self.handle_bridge_mode()

    def _ask_resolution(self):
        print("Разрешение:")
        print("1. 720p (Быстро, меньше квоты)")
        print("2. 1080p (HD)")
        print("3. 4k (UHD)") 
        c = input("Выбор (1): ")
        if c == "2": return "1080p"
        if c == "3": return "4k"
        return "720p"

    def _configure_advanced_options(self, prompt):
        """Ask for negative prompt, references, etc."""
        # 1. Negative Prompt
        neg = input("Negative Prompt (Enter для авто-генерации): ").strip()
        if not neg:
            print("Генерация негативного промпта...")
            neg = self.ai.generate_negative_prompt(prompt)
            print(f"Использован негативный промпт: {neg}")
        
        # 2. Reference Images
        refs = []
        c_ref = input("Добавить reference images? (y/n): ").lower()
        if c_ref == 'y':
            print("Введите пути к изображениям (до 3), пустая строка для завершения:")
            while len(refs) < 3:
                p_ref = input(f"Image {len(refs)+1}: ").strip('"').strip()
                if not p_ref: break
                if os.path.exists(p_ref):
                    refs.append(p_ref)
                else:
                    print("Файл не найден.")

        # 3. Last Frame (Constraint)
        last_frame = None
        c_last = input("Задать Last Frame (концовку)? (1-Файл, 2-Сгенерировать, Enter-Нет): ")
        if c_last == "1":
            p_last = input("Путь к файлу: ").strip('"').strip()
            if os.path.exists(p_last):
                last_frame = p_last
        elif c_last == "2":
            p_gen = input("Промпт для генерации концовки: ")
            fname = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "gen_last_frame", ".png")
            last_frame = self.ai.generate_image(p_gen, fname)
            
        return neg, refs, last_frame

    def handle_single_video(self):
        prompt = input("Промпт: ")
        fname = FileManager.ask_for_filename("Имя файла (Enter для авто): ", "veo_video", Config.OUTPUT_DIRS["video"], ".mp4")
        ar = input("Формат (1 - 16:9, 2 - 9:16): ")
        ratio = "9:16" if ar == "2" else "16:9"
        res = self._ask_resolution()
        
        neg, refs, last_frame = self._configure_advanced_options(prompt)
        
        self.ai.generate_video(prompt, fname, aspect_ratio=ratio, resolution=res, negative_prompt=neg, reference_image_paths=refs, last_frame_path=last_frame)

    def handle_image_gen(self):
        p = input("Промпт: ")
        fname = FileManager.ask_for_filename("Имя файла: ", "generated_image", Config.OUTPUT_DIRS["image"], ".png")
        self.ai.generate_image(p, fname)

    def handle_audio_gen(self):
        p = input("Промпт: ")
        sec = 10
        try: sec = int(input("Сек (10): "))
        except: pass
        fname = FileManager.ask_for_filename("Имя файла: ", "audio", Config.OUTPUT_DIRS["audio"], ".wav")
        self.ai.generate_audio(p, sec, fname)

    def handle_batch_mode(self):
        print("1. Из списка промптов\n2. Склейка готовых видео")
        sub = input("Выбор: ")
        
        if sub == "1":
            raw = input("Промпты через ';': ")
            prompts = [x.strip() for x in raw.split(';') if x.strip()]
            res = self._ask_resolution()
            for i, p in enumerate(prompts):
                fname = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], f"batch_{i+1}", ".mp4")
                self.ai.generate_video(p, fname, resolution=res)
        
        elif sub == "2":
            print(f"Видео должны быть в папке: {Config.OUTPUT_DIRS['video']}")
            names = input("Имена файлов (без .mp4) через запятую: ")
            files = []
            for n in names.split(','):
                n = n.strip()
                path = os.path.join(Config.OUTPUT_DIRS["video"], n + ".mp4")
                if os.path.exists(path):
                    files.append(path)
                else:
                    print(f"Не найден: {n}")
            
            if len(files) > 1:
                out = FileManager.ask_for_filename("Имя итогового файла: ", "batch_concat", Config.OUTPUT_DIRS["video"], ".mp4")
                MediaProcessor.concatenate_videos(files, out)

    def handle_chain_mode(self):
        print("\n--- Режим цепочки ---")
        ar = input("Формат для всей цепочки (1 - 16:9, 2 - 9:16): ")
        ratio = "9:16" if ar == "2" else "16:9"
        res = self._ask_resolution()

        start_opt = input("Начало: 1-Текст, 2-Сгенер. фото, 3-Свой файл: ")
        
        video_chain = []
        temp_files = [] 
        
        first_vid_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "chain_segment_1", ".mp4")
        
        if start_opt == "1":
            p = input("Промпт для 1 видео: ")
            neg, refs, last_frame = self._configure_advanced_options(p)
            v = self.ai.generate_video(p, first_vid_name, aspect_ratio=ratio, resolution=res, negative_prompt=neg, reference_image_paths=refs, last_frame_path=last_frame)
            if v: 
                video_chain.append(v)
                temp_files.append(v)
        elif start_opt == "2":
            p = input("Промпт для фото: ")
            img_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "chain_start_gen", ".png")
            img = self.ai.generate_image(p, img_name)
            if img:
                temp_files.append(img)
                p_vid = input("Промпт для видео: ")
                neg, refs, last_frame = self._configure_advanced_options(p_vid)
                v = self.ai.generate_video(p_vid, first_vid_name, image_path=img, aspect_ratio=ratio, resolution=res, negative_prompt=neg, reference_image_paths=refs, last_frame_path=last_frame)
                if v:
                    video_chain.append(v)
                    temp_files.append(v)
        elif start_opt == "3":
            path = input("Путь к фото: ").strip('"')
            if os.path.exists(path):
                p_vid = input("Промпт для видео: ")
                neg, refs, last_frame = self._configure_advanced_options(p_vid)
                v = self.ai.generate_video(p_vid, first_vid_name, image_path=path, aspect_ratio=ratio, resolution=res, negative_prompt=neg, reference_image_paths=refs, last_frame_path=last_frame)
                if v:
                    video_chain.append(v)
                    temp_files.append(v)

        if video_chain:
            try: n = int(input("Сколько еще звеньев добавить? "))
            except: n = 0
            
            for i in range(n):
                print(f"\n--- Звено {i+2} ---")
                last_vid = video_chain[-1]
                last_frame_path = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], f"temp_frame_{i}", ".png")
                
                if MediaProcessor.extract_frame(last_vid, last_frame_path, "last"):
                    temp_files.append(last_frame_path)
                    prompt = input("Промпт (Enter = продолжить действие): ")
                    if not prompt: prompt = "Continue the action, high quality"
                    
                    # For subsequent links, we might want fewer advanced options or ask again?
                    # Let's keep it simple: Just negative prompt auto-gen or same? 
                    # User asked for features generally. Let's allow full config.
                    neg, refs, last_frame = self._configure_advanced_options(prompt)

                    next_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], f"chain_segment_{i+2}", ".mp4")
                    v_next = self.ai.generate_video(prompt, next_name, image_path=last_frame_path, aspect_ratio=ratio, resolution=res, negative_prompt=neg, reference_image_paths=refs, last_frame_path=last_frame)
                    if v_next:
                        video_chain.append(v_next)
                        temp_files.append(v_next)
                    else: break
                else: break
            
            if len(video_chain) > 0:
                final_name = FileManager.ask_for_filename("Имя итогового видео: ", "full_chain", Config.OUTPUT_DIRS["video"], ".mp4")
                if len(video_chain) > 1:
                    MediaProcessor.concatenate_videos(video_chain, final_name)
                else:
                    shutil.copy(video_chain[0], final_name)
                    print(f"Видео сохранено: {final_name}")
                FileManager.cleanup_files(temp_files)
        else:
            print("Цепочка прервана.")

    def handle_bridge_mode(self):
        print("\n--- Interpolation / Bridge with Verification ---")
        print(f"Укажите название видео 1 (Начало) из папки {Config.OUTPUT_DIRS['video']}")
        n1 = input("Видео 1: ").strip()
        p1 = os.path.join(Config.OUTPUT_DIRS["video"], n1 + ".mp4") if not n1.endswith(".mp4") else os.path.join(Config.OUTPUT_DIRS["video"], n1)
        if not os.path.exists(p1):
            print("Видео 1 не найдено.")
            return

        ar = input("Формат (1 - 16:9, 2 - 9:16): ")
        ratio = "9:16" if ar == "2" else "16:9"
        res = self._ask_resolution()

        print("2. Видео 2 (Конец):\n   1. Выбрать существующее видео\n   2. Сгенерировать новое видео")
        v2_choice = input("Выбор: ")
        
        p2 = None
        is_v2_generated = False
        temp_files = []

        if v2_choice == "1":
            n2 = input("Видео 2: ").strip()
            p2 = os.path.join(Config.OUTPUT_DIRS["video"], n2 + ".mp4") if not n2.endswith(".mp4") else os.path.join(Config.OUTPUT_DIRS["video"], n2)
            if not os.path.exists(p2):
                print("Видео 2 не найдено.")
                return
        elif v2_choice == "2":
            prompt_v2 = input("Промпт для Видео 2: ")
            p2 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "temp_v2_initial", ".mp4")
            if self.ai.generate_video(prompt_v2, p2, aspect_ratio=ratio, resolution=res):
                is_v2_generated = True
                temp_files.append(p2)
            else: return
        else: return

        print("\n--- Генерация Моста ---")
        f1 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "bridge_start", ".png")
        f2 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "bridge_end", ".png")
        temp_files.extend([f1, f2])

        if not (MediaProcessor.extract_frame(p1, f1, "last") and MediaProcessor.extract_frame(p2, f2, "first")):
            print("Ошибка извлечения кадров.")
            return

        prompt_bridge = input("Промпт перехода (Enter для авто): ")
        if not prompt_bridge: prompt_bridge = "Smooth transition between the two scenes, cinematic"
        
        bridge_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "temp_bridge", ".mp4")
        if not self.ai.generate_video(prompt_bridge, bridge_name, image_path=f1, last_frame_path=f2, aspect_ratio=ratio, resolution=res):
            return
        
        temp_files.append(bridge_name)
        current_v2 = p2
        
        while True:
            preview_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "preview_stitched", ".mp4")
            MediaProcessor.concatenate_videos([p1, bridge_name, current_v2], preview_name)
            
            print(f"\nГотово! Проверьте видео: {preview_name}")
            print("Довольны результатом?")
            print("1. Да, сохранить")
            print("2. Нет, перегенерировать Видео 2 (используя конец моста)")
            
            dec = input("Выбор: ")
            if dec == "1":
                final_name = FileManager.ask_for_filename("Имя итогового файла: ", "final_bridged_video", Config.OUTPUT_DIRS["video"], ".mp4")
                try:
                    shutil.move(preview_name, final_name)
                    print(f"Сохранено как {final_name}")
                except Exception as e:
                    print(f"Ошибка сохранения: {e}")
                    shutil.copy(preview_name, final_name)
                
                FileManager.cleanup_files(temp_files)
                if os.path.exists(preview_name): os.remove(preview_name)
                if is_v2_generated and os.path.exists(current_v2):
                    try: os.remove(current_v2)
                    except: pass
                break
            elif dec == "2":
                print("\n--- Регенерация Видео 2 ---")
                if is_v2_generated and os.path.exists(current_v2):
                    try: os.remove(current_v2)
                    except: pass
                
                bridge_end_frame = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "bridge_last_frame", ".png")
                temp_files.append(bridge_end_frame)
                
                if MediaProcessor.extract_frame(bridge_name, bridge_end_frame, "last"):
                    p_new = input("Промпт для нового Видео 2: ")
                    new_v2_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "temp_v2_regen", ".mp4")
                    if self.ai.generate_video(p_new, new_v2_name, image_path=bridge_end_frame, aspect_ratio=ratio, resolution=res):
                        current_v2 = new_v2_name
                        is_v2_generated = True
                    else:
                        print("Ошибка генерации. Оставляем предыдущее видео.")
                else:
                    print("Ошибка извлечения кадра из моста.")
                
                if os.path.exists(preview_name):
                    try: os.remove(preview_name)
                    except: pass
            else:
                print("Неверный ввод.")

if __name__ == "__main__":
    app = VeoGeneratorApp()
    app.run()
