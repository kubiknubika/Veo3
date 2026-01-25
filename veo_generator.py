import os
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

# Initialize client (global for Veo/Imagen)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- Directory Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRS = {
    "video": os.path.join(BASE_DIR, "output", "videos"),
    "image": os.path.join(BASE_DIR, "output", "images"),
    "audio": os.path.join(BASE_DIR, "output", "audio"),
}

# Ensure output directories exist
for path in OUTPUT_DIRS.values():
    os.makedirs(path, exist_ok=True)

# --- Helper Functions ---

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

def ask_for_filename(prompt_text, default_name, folder, extension):
    name = input(prompt_text)
    if not name.strip():
        name = default_name
    return get_unique_filename(folder, name, extension, strict=True)

# --- Core Functions ---

def generate_audio_lyria(prompt, duration_seconds=10, output_filename=None):
    """Generates audio using Lyria RealTime."""
    if not output_filename:
        output_filename = get_unique_filename(OUTPUT_DIRS["audio"], "generated_music", ".wav")
        
    print(f"Генерация музыки: '{prompt}' ({duration_seconds} сек)...")
    
    async def run_audio_session():
        client_lyria = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options={'api_version': 'v1alpha'}
        )
        
        audio_buffer = bytearray()
        model = 'models/lyria-realtime-exp'
        
        try:
            async with client_lyria.aio.live.music.connect(model=model) as session:
                await session.set_weighted_prompts(
                  prompts=[types.WeightedPrompt(text=prompt, weight=1.0)]
                )
                await session.set_music_generation_config(
                  config=types.LiveMusicGenerationConfig(bpm=120, temperature=1.0)
                )

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

def get_image_param(image_path):
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

def generate_video(prompt, output_filename, model="veo-3.1-generate-preview", image_path=None, last_frame_path=None, aspect_ratio="16:9"):
    print(f"Генерация видео: '{prompt}'")
    
    try:
        config_params = {"aspect_ratio": aspect_ratio}
        
        image_param = None
        if image_path:
            image_param = get_image_param(image_path)
            if not image_param: return None

        if last_frame_path:
            last_frame_param = get_image_param(last_frame_path)
            if last_frame_param:
                config_params["last_frame"] = last_frame_param

        config = types.GenerateVideosConfig(**config_params)

        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image_param,
            config=config
        )

        print("Ожидание генерации...")
        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)

        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(output_filename)
        print(f"Видео сохранено: {output_filename}")
        return output_filename

    except Exception as e:
        print(f"Ошибка генерации видео: {e}")
        return None

def generate_image(prompt, output_filename):
    print(f"Генерация изображения: '{prompt}'")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
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

# --- Main Logic ---

def main():
    while True:
        print("\n=== Veo3 Pro Generator ===")
        print("1. Одиночное видео (Text-to-Video)")
        print("2. Цепочка видео (Склейка)")
        print("3. Генерация изображения (Text-to-Image)")
        print("4. Батч режим")
        print("5. Генерация аудио (Lyria)")
        print("6. Мост между видео (Interpolation)")
        print("0. Выход")
        
        choice = input("Выберите режим: ")

        if choice == "0":
            break

        elif choice == "1":
            prompt = input("Промпт: ")
            fname = ask_for_filename("Имя файла (Enter для авто): ", "veo_video", OUTPUT_DIRS["video"], ".mp4")
            ar = input("Формат (1 - 16:9, 2 - 9:16): ")
            ratio = "9:16" if ar == "2" else "16:9"
            generate_video(prompt, fname, aspect_ratio=ratio)

        elif choice == "2":
            print("\n--- Режим цепочки ---")
            ar = input("Формат для всей цепочки (1 - 16:9, 2 - 9:16): ")
            ratio = "9:16" if ar == "2" else "16:9"

            start_opt = input("Начало: 1-Текст, 2-Сгенер. фото, 3-Свой файл: ")
            
            video_chain = []
            temp_files = [] # Files to delete later
            
            # Step 1: First Video
            first_vid_name = get_unique_filename(OUTPUT_DIRS["video"], "chain_segment_1", ".mp4")
            
            if start_opt == "1":
                p = input("Промпт для 1 видео: ")
                v = generate_video(p, first_vid_name, aspect_ratio=ratio)
                if v: 
                    video_chain.append(v)
                    temp_files.append(v)

            elif start_opt == "2":
                p = input("Промпт для фото: ")
                img_name = get_unique_filename(OUTPUT_DIRS["image"], "chain_start_gen", ".png")
                img = generate_image(p, img_name)
                if img:
                    temp_files.append(img) # Mark generated image for deletion
                    p_vid = input("Промпт для видео: ")
                    v = generate_video(p_vid, first_vid_name, image_path=img, aspect_ratio=ratio)
                    if v:
                        video_chain.append(v)
                        temp_files.append(v)

            elif start_opt == "3":
                path = input("Путь к фото: ").strip('"')
                if os.path.exists(path):
                    p_vid = input("Промпт для видео: ")
                    v = generate_video(p_vid, first_vid_name, image_path=path, aspect_ratio=ratio)
                    if v:
                        video_chain.append(v)
                        temp_files.append(v)
                else:
                    print("Файл не найден.")

            # Step 2: Continuation
            if video_chain:
                while True:
                    try:
                        n = int(input("Сколько еще звеньев добавить? "))
                        break
                    except: pass
                
                for i in range(n):
                    print(f"\n--- Звено {i+2} ---")
                    last_vid = video_chain[-1]
                    last_frame_path = get_unique_filename(OUTPUT_DIRS["image"], f"temp_frame_{i}", ".png")
                    
                    if extract_frame(last_vid, last_frame_path, "last"):
                        temp_files.append(last_frame_path)
                        
                        prompt = input("Промпт (Enter = продолжить действие): ")
                        if not prompt: prompt = "Continue the action, high quality"
                        
                        next_name = get_unique_filename(OUTPUT_DIRS["video"], f"chain_segment_{i+2}", ".mp4")
                        v_next = generate_video(prompt, next_name, image_path=last_frame_path, aspect_ratio=ratio)
                        
                        if v_next:
                            video_chain.append(v_next)
                            temp_files.append(v_next)
                        else:
                            break
                    else:
                        break
                
                # Step 3: Concatenate and Clean
                if len(video_chain) > 0:
                    final_name = ask_for_filename("Имя итогового видео: ", "full_chain", OUTPUT_DIRS["video"], ".mp4")
                    if len(video_chain) > 1:
                        concatenate_videos(video_chain, final_name)
                    else:
                        # Just rename/copy the single video
                        shutil.copy(video_chain[0], final_name)
                        print(f"Видео сохранено: {final_name}")

                    # Cleanup
                    print("Очистка временных файлов...")
                    for f in temp_files:
                        if os.path.exists(f):
                            try:
                                os.remove(f)
                            except Exception as e:
                                print(f"Не удалось удалить {f}: {e}")
            else:
                print("Цепочка прервана в начале.")

        elif choice == "3":
            p = input("Промпт: ")
            fname = ask_for_filename("Имя файла: ", "generated_image", OUTPUT_DIRS["image"], ".png")
            generate_image(p, fname)

        elif choice == "4":
            print("1. Из списка промптов\n2. Склейка готовых видео")
            sub = input("Выбор: ")
            
            if sub == "1":
                raw = input("Промпты через ';': ")
                prompts = [x.strip() for x in raw.split(';') if x.strip()]
                for i, p in enumerate(prompts):
                    fname = get_unique_filename(OUTPUT_DIRS["video"], f"batch_{i+1}", ".mp4")
                    generate_video(p, fname)
            
            elif sub == "2":
                print(f"Видео должны быть в папке: {OUTPUT_DIRS['video']}")
                names = input("Имена файлов (без .mp4) через запятую: ")
                files = []
                for n in names.split(','):
                    n = n.strip()
                    path = os.path.join(OUTPUT_DIRS["video"], n + ".mp4")
                    if os.path.exists(path):
                        files.append(path)
                    else:
                        print(f"Не найден: {n}")
                
                if len(files) > 1:
                    out = ask_for_filename("Имя итогового файла: ", "batch_concat", OUTPUT_DIRS["video"], ".mp4")
                    concatenate_videos(files, out)

        elif choice == "5":
            p = input("Промпт: ")
            sec = 10
            try: sec = int(input("Сек (10): "))
            except: pass
            fname = ask_for_filename("Имя файла: ", "audio", OUTPUT_DIRS["audio"], ".wav")
            generate_audio_lyria(p, sec, fname)

        elif choice == "6":
            print("\n--- Interpolation / Bridge with Verification ---")
            
            # 1. Select V1
            print(f"Укажите название видео 1 (Начало) из папки {OUTPUT_DIRS['video']}")
            n1 = input("Видео 1: ").strip()
            p1 = os.path.join(OUTPUT_DIRS["video"], n1 + ".mp4") if not n1.endswith(".mp4") else os.path.join(OUTPUT_DIRS["video"], n1)
            if not os.path.exists(p1):
                print("Видео 1 не найдено.")
                continue

            # Aspect Ratio
            ar = input("Формат (1 - 16:9, 2 - 9:16): ")
            ratio = "9:16" if ar == "2" else "16:9"

            # 2. Select V2 (Initial)
            print("2. Видео 2 (Конец):")
            print("   1. Выбрать существующее видео")
            print("   2. Сгенерировать новое видео")
            v2_choice = input("Выбор: ")
            
            p2 = None
            is_v2_generated = False
            temp_files = [] # Track temps to clean

            if v2_choice == "1":
                n2 = input("Видео 2: ").strip()
                p2 = os.path.join(OUTPUT_DIRS["video"], n2 + ".mp4") if not n2.endswith(".mp4") else os.path.join(OUTPUT_DIRS["video"], n2)
                if not os.path.exists(p2):
                    print("Видео 2 не найдено.")
                    continue
            elif v2_choice == "2":
                prompt_v2 = input("Промпт для Видео 2: ")
                p2 = get_unique_filename(OUTPUT_DIRS["video"], "temp_v2_initial", ".mp4")
                if generate_video(prompt_v2, p2, aspect_ratio=ratio):
                    is_v2_generated = True
                    temp_files.append(p2)
                else:
                    continue
            else:
                continue

            # 3. Generate Bridge (V1 -> V2)
            print("\n--- Генерация Моста ---")
            f1 = get_unique_filename(OUTPUT_DIRS["image"], "bridge_start", ".png")
            f2 = get_unique_filename(OUTPUT_DIRS["image"], "bridge_end", ".png")
            temp_files.extend([f1, f2])

            if not (extract_frame(p1, f1, "last") and extract_frame(p2, f2, "first")):
                print("Ошибка извлечения кадров.")
                continue 

            prompt_bridge = input("Промпт перехода (Enter для авто): ")
            if not prompt_bridge: prompt_bridge = "Smooth transition between the two scenes, cinematic"
            
            bridge_name = get_unique_filename(OUTPUT_DIRS["video"], "temp_bridge", ".mp4")
            if not generate_video(prompt_bridge, bridge_name, image_path=f1, last_frame_path=f2, aspect_ratio=ratio):
                continue
            
            temp_files.append(bridge_name)

            # 4. The Loop
            current_v2 = p2
            
            while True:
                # Stitch [V1, Bridge, current_v2]
                preview_name = get_unique_filename(OUTPUT_DIRS["video"], "preview_stitched", ".mp4")
                concatenate_videos([p1, bridge_name, current_v2], preview_name)
                
                print(f"\nГотово! Проверьте видео: {preview_name}")
                print("Довольны результатом?")
                print("1. Да, сохранить (мост и темп. файлы удалятся)")
                print("2. Нет, перегенерировать Видео 2 (используя конец моста)")
                
                dec = input("Выбор: ")
                
                if dec == "1":
                    final_name = ask_for_filename("Имя итогового файла: ", "final_bridged_video", OUTPUT_DIRS["video"], ".mp4")
                    try:
                        shutil.move(preview_name, final_name)
                        print(f"Сохранено как {final_name}")
                    except Exception as e:
                        print(f"Ошибка сохранения: {e}")
                        shutil.copy(preview_name, final_name)
                    
                    # Cleanup
                    print("Очистка временных файлов...")
                    if os.path.exists(preview_name): os.remove(preview_name)
                    for f in temp_files:
                        if os.path.exists(f): 
                            try: os.remove(f)
                            except: pass
                    
                    # Delete the NEW generated V2 if we made one (since it's baked into the final file now)
                    if is_v2_generated and os.path.exists(current_v2):
                         try: os.remove(current_v2)
                         except: pass
                    
                    break

                elif dec == "2":
                    print("\n--- Регенерация Видео 2 ---")
                    
                    # Clean up OLD generated V2 (if it was generated)
                    if is_v2_generated and os.path.exists(current_v2):
                        try: os.remove(current_v2)
                        except: pass
                    
                    # Extract last frame of Bridge
                    bridge_end_frame = get_unique_filename(OUTPUT_DIRS["image"], "bridge_last_frame", ".png")
                    temp_files.append(bridge_end_frame)
                    
                    if extract_frame(bridge_name, bridge_end_frame, "last"):
                        p_new = input("Промпт для нового Видео 2: ")
                        new_v2_name = get_unique_filename(OUTPUT_DIRS["video"], "temp_v2_regen", ".mp4")
                        
                        if generate_video(p_new, new_v2_name, image_path=bridge_end_frame, aspect_ratio=ratio):
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
    main()
