import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
import shutil
from veo_generator import GenAIService, Config, FileManager, MediaProcessor

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")
    
    def flush(self):
        pass

class VeoGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Veo3 Pro Generator UI")
        self.root.geometry("900x700")
        
        self.ai = GenAIService()
        Config.ensure_dirs()
        
        # Bridge Mode State
        self.bridge_state = {
            "p1": None,
            "bridge": None,
            "v2_type": None, # 'existing' or 'generated'
            "v2_path": None,
            "preview": None,
            "temp_files": []
        }

        self._init_ui()

    def _init_ui(self):
        # Main Container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Output Console (Bottom)
        console_frame = ttk.LabelFrame(main_frame, text="Log Output", height=200)
        console_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=10, state="disabled", font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout
        sys.stdout = TextRedirector(self.console, "stdout")
        sys.stderr = TextRedirector(self.console, "stderr")

        # Tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Single Generation
        self.tab_single = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_single, text="Single Gen")
        self._setup_single_tab()

        # Tab 2: Bridge/Interpolation
        self.tab_bridge = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bridge, text="Bridge / Interpolation")
        self._setup_bridge_tab()
        
        # Tab 3: Utils (Image/Audio)
        self.tab_utils = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_utils, text="Image & Audio")
        self._setup_utils_tab()

    def _setup_single_tab(self):
        frame = ttk.Frame(self.tab_single, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Prompt
        ttk.Label(frame, text="Prompt:").pack(anchor=tk.W)
        self.single_prompt = tk.Text(frame, height=5, width=60)
        self.single_prompt.pack(fill=tk.X, pady=5)

        # Options Grid
        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=10)

        ttk.Label(opts, text="Resolution:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.res_var = tk.StringVar(value="720p")
        ttk.Combobox(opts, textvariable=self.res_var, values=["720p", "1080p", "4k"], state="readonly").grid(row=0, column=1, padx=5)

        ttk.Label(opts, text="Aspect Ratio:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.ar_var = tk.StringVar(value="16:9")
        ttk.Combobox(opts, textvariable=self.ar_var, values=["16:9", "9:16"], state="readonly").grid(row=0, column=3, padx=5)

        # Negative Prompt
        ttk.Label(frame, text="Negative Prompt (Auto if empty):").pack(anchor=tk.W)
        self.neg_prompt = tk.Entry(frame)
        self.neg_prompt.pack(fill=tk.X, pady=5)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(btn_frame, text="Generate Video", command=self._run_single_gen).pack(side=tk.LEFT)

    def _setup_utils_tab(self):
        frame = ttk.Frame(self.tab_utils, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Image Gen
        lf_img = ttk.LabelFrame(frame, text="Image Generation", padding=10)
        lf_img.pack(fill=tk.X, pady=10)
        
        ttk.Label(lf_img, text="Prompt:").pack(anchor=tk.W)
        self.img_prompt = tk.Entry(lf_img)
        self.img_prompt.pack(fill=tk.X, pady=5)
        ttk.Button(lf_img, text="Generate Image", command=self._run_image_gen).pack(anchor=tk.E)

        # Audio Gen
        lf_aud = ttk.LabelFrame(frame, text="Audio Generation (Lyria)", padding=10)
        lf_aud.pack(fill=tk.X, pady=10)
        
        ttk.Label(lf_aud, text="Prompt:").pack(anchor=tk.W)
        self.aud_prompt = tk.Entry(lf_aud)
        self.aud_prompt.pack(fill=tk.X, pady=5)
        
        dur_frame = ttk.Frame(lf_aud)
        dur_frame.pack(fill=tk.X)
        ttk.Label(dur_frame, text="Duration (sec):").pack(side=tk.LEFT)
        self.aud_dur = tk.Entry(dur_frame, width=10)
        self.aud_dur.insert(0, "10")
        self.aud_dur.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(lf_aud, text="Generate Audio", command=self._run_audio_gen).pack(anchor=tk.E)

    def _setup_bridge_tab(self):
        frame = ttk.Frame(self.tab_bridge, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Video 1 Selection
        v1_frame = ttk.Frame(frame)
        v1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(v1_frame, text="Video 1 (Start):").pack(side=tk.LEFT)
        self.v1_path_var = tk.StringVar()
        ttk.Entry(v1_frame, textvariable=self.v1_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(v1_frame, text="Browse", command=lambda: self._browse_file(self.v1_path_var)).pack(side=tk.LEFT)

        # Video 2 Selection
        v2_frame = ttk.Frame(frame)
        v2_frame.pack(fill=tk.X, pady=5)
        ttk.Label(v2_frame, text="Video 2 (End):").pack(side=tk.LEFT)
        self.v2_path_var = tk.StringVar()
        ttk.Entry(v2_frame, textvariable=self.v2_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(v2_frame, text="Browse (Optional)", command=lambda: self._browse_file(self.v2_path_var)).pack(side=tk.LEFT)
        ttk.Label(frame, text="* Leave Video 2 empty to generate it from prompt").pack(anchor=tk.W, padx=5)

        # Bridge Prompt
        ttk.Label(frame, text="Bridge Transition Prompt:").pack(anchor=tk.W, pady=(10,0))
        self.bridge_prompt = tk.Entry(frame)
        self.bridge_prompt.pack(fill=tk.X, pady=5)

        # Video 2 Prompt (if generated)
        ttk.Label(frame, text="Video 2 Prompt (if generating new):").pack(anchor=tk.W, pady=(10,0))
        self.v2_gen_prompt = tk.Entry(frame)
        self.v2_gen_prompt.pack(fill=tk.X, pady=5)

        # Bridge Actions
        act_frame = ttk.Frame(frame)
        act_frame.pack(fill=tk.X, pady=20)
        
        self.btn_start_bridge = ttk.Button(act_frame, text="Start Bridge Process", command=self._start_bridge_thread)
        self.btn_start_bridge.pack(side=tk.LEFT)

        # Verification Controls (Hidden initially)
        self.verif_frame = ttk.LabelFrame(frame, text="Verification Loop", padding=10)
        self.verif_frame.pack(fill=tk.X, pady=20)
        
        self.lbl_status = ttk.Label(self.verif_frame, text="Waiting for generation...")
        self.lbl_status.pack(fill=tk.X)
        
        btn_verif = ttk.Frame(self.verif_frame)
        btn_verif.pack(fill=tk.X, pady=5)
        
        self.btn_accept = ttk.Button(btn_verif, text="✅ Accept & Save", command=self._bridge_accept, state="disabled")
        self.btn_accept.pack(side=tk.LEFT, padx=5)
        
        self.btn_regen = ttk.Button(btn_verif, text="🔄 Regenerate V2", command=self._bridge_regen, state="disabled")
        self.btn_regen.pack(side=tk.LEFT, padx=5)

    def _browse_file(self, var):
        f = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if f: var.set(f)

    def _run_in_thread(self, target):
        threading.Thread(target=target, daemon=True).start()

    # --- Actions ---

    def _run_single_gen(self):
        prompt = self.single_prompt.get("1.0", tk.END).strip()
        if not prompt: return
        
        def task():
            print(f"Starting Video Generation...")
            neg = self.neg_prompt.get().strip() or None
            if not neg:
                print("Auto-generating negative prompt...")
                neg = self.ai.generate_negative_prompt(prompt)
                print(f"Negative: {neg}")

            fname = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "ui_video", ".mp4")
            self.ai.generate_video(
                prompt, fname, 
                aspect_ratio=self.ar_var.get(), 
                resolution=self.res_var.get(),
                negative_prompt=neg
            )
            print("Done!")
        
        self._run_in_thread(task)

    def _run_image_gen(self):
        prompt = self.img_prompt.get().strip()
        if not prompt: return
        
        # Ask for file in main thread to avoid threading issues with dialogs
        fname = filedialog.asksaveasfilename(
            initialdir=Config.OUTPUT_DIRS["image"],
            initialfile="ui_image.png",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if not fname: return

        def task():
            self.ai.generate_image(prompt, fname)
        self._run_in_thread(task)

    def _run_audio_gen(self):
        prompt = self.aud_prompt.get().strip()
        try: dur = int(self.aud_dur.get())
        except: dur = 10
        if not prompt: return

        fname = filedialog.asksaveasfilename(
            initialdir=Config.OUTPUT_DIRS["audio"],
            initialfile="ui_audio.wav",
            defaultextension=".wav",
            filetypes=[("WAV Audio", "*.wav")]
        )
        if not fname: return

        def task():
            self.ai.generate_audio(prompt, dur, fname)
        self._run_in_thread(task)

    # --- Bridge Logic ---

    def _start_bridge_thread(self):
        self.btn_start_bridge.configure(state="disabled")
        self.btn_accept.configure(state="disabled")
        self.btn_regen.configure(state="disabled")
        self._run_in_thread(self._process_bridge_initial)

    def _process_bridge_initial(self):
        # 1. Validate V1
        v1 = self.v1_path_var.get().strip()
        if not os.path.exists(v1):
            print("Error: Video 1 not found")
            self.root.after(0, lambda: self.btn_start_bridge.configure(state="normal"))
            return
        
        self.bridge_state["p1"] = v1
        self.bridge_state["temp_files"] = []

        # 2. Determine V2
        v2_input = self.v2_path_var.get().strip()
        if v2_input and os.path.exists(v2_input):
            self.bridge_state["v2_type"] = "existing"
            self.bridge_state["v2_path"] = v2_input
        else:
            self.bridge_state["v2_type"] = "generated"
            prompt_v2 = self.v2_gen_prompt.get().strip()
            if not prompt_v2:
                print("Error: Provide Video 2 path OR Prompt")
                self.root.after(0, lambda: self.btn_start_bridge.configure(state="normal"))
                return
            
            print("Generating Initial Video 2...")
            v2_path = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "ui_bridge_v2", ".mp4")
            if self.ai.generate_video(prompt_v2, v2_path, resolution=self.res_var.get()): # Use simple params for V2 initial
                self.bridge_state["v2_path"] = v2_path
                self.bridge_state["temp_files"].append(v2_path)
            else:
                self.root.after(0, lambda: self.btn_start_bridge.configure(state="normal"))
                return

        # 3. Generate Bridge
        print("Extracting frames...")
        f1 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "ui_br_start", ".png")
        f2 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "ui_br_end", ".png")
        self.bridge_state["temp_files"].extend([f1, f2])

        if MediaProcessor.extract_frame(self.bridge_state["p1"], f1, "last") and \
           MediaProcessor.extract_frame(self.bridge_state["v2_path"], f2, "first"):
            
            print("Generating Bridge Video...")
            b_prompt = self.bridge_prompt.get().strip() or "Smooth transition"
            b_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "ui_bridge_seg", ".mp4")
            
            if self.ai.generate_video(b_prompt, b_name, image_path=f1, last_frame_path=f2, resolution=self.res_var.get()):
                self.bridge_state["bridge"] = b_name
                self.bridge_state["temp_files"].append(b_name)
                
                # Stitch Preview
                self._stitch_preview()
            else:
                print("Bridge generation failed.")
                self.root.after(0, lambda: self.btn_start_bridge.configure(state="normal"))
        else:
            print("Frame extraction failed.")
            self.root.after(0, lambda: self.btn_start_bridge.configure(state="normal"))

    def _stitch_preview(self):
        print("Stitching preview...")
        prev_name = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "ui_preview_stitch", ".mp4")
        files = [self.bridge_state["p1"], self.bridge_state["bridge"], self.bridge_state["v2_path"]]
        
        if MediaProcessor.concatenate_videos(files, prev_name):
            self.bridge_state["preview"] = prev_name
            print(f"Preview Ready: {prev_name}")
            
            # Update UI
            self.root.after(0, self._enable_verif_ui)
        else:
            print("Stitching failed.")

    def _enable_verif_ui(self):
        self.lbl_status.configure(text=f"Preview created. Check: {os.path.basename(self.bridge_state['preview'])}")
        self.btn_accept.configure(state="normal")
        
        # Only allow regen if V2 is generated type (or user wants to replace existing?)
        # Logic matches CLI: only if we are generating V2 can we regen it easily based on bridge end.
        # Actually CLI deletes current V2.
        self.btn_regen.configure(state="normal") 

    def _bridge_accept(self):
        # Save
        dest = filedialog.asksaveasfilename(defaultextension=".mp4", initialdir=Config.OUTPUT_DIRS["video"], initialfile="final_bridge.mp4")
        if dest:
            try:
                shutil.copy(self.bridge_state["preview"], dest)
                print(f"Saved to {dest}")
                
                # Cleanup
                if self.bridge_state["preview"]: os.remove(self.bridge_state["preview"])
                FileManager.cleanup_files(self.bridge_state["temp_files"])
                
                # If V2 was generated, we delete the temp file because it's baked into final
                # But wait, maybe user wants it? Config says "delete generated V2" in CLI.
                # Let's clean it up to be safe.
                
                self.btn_start_bridge.configure(state="normal")
                self.btn_accept.configure(state="disabled")
                self.btn_regen.configure(state="disabled")
                self.lbl_status.configure(text="Saved.")
            except Exception as e:
                print(f"Error saving: {e}")

    def _bridge_regen(self):
        prompt = self.v2_gen_prompt.get().strip()
        if not prompt:
            print("Please enter a Video 2 prompt for regeneration.")
            return
        
        self.btn_accept.configure(state="disabled")
        self.btn_regen.configure(state="disabled")
        self.lbl_status.configure(text="Regenerating V2 from bridge end...")
        
        self._run_in_thread(lambda: self._process_regen(prompt))

    def _process_regen(self, prompt):
        # 1. Cleanup old preview and old V2 (if generated)
        if self.bridge_state["preview"] and os.path.exists(self.bridge_state["preview"]):
            os.remove(self.bridge_state["preview"])
        
        if self.bridge_state["v2_type"] == "generated" and self.bridge_state["v2_path"] and os.path.exists(self.bridge_state["v2_path"]):
            try: os.remove(self.bridge_state["v2_path"])
            except: pass
            
        # 2. Extract last frame of Bridge
        bridge_end_frame = FileManager.get_unique_filename(Config.OUTPUT_DIRS["image"], "ui_br_last", ".png")
        self.bridge_state["temp_files"].append(bridge_end_frame)
        
        if MediaProcessor.extract_frame(self.bridge_state["bridge"], bridge_end_frame, "last"):
            # 3. Generate New V2
            new_v2 = FileManager.get_unique_filename(Config.OUTPUT_DIRS["video"], "ui_v2_regen", ".mp4")
            print("Generating New V2...")
            if self.ai.generate_video(prompt, new_v2, image_path=bridge_end_frame, resolution=self.res_var.get()):
                self.bridge_state["v2_path"] = new_v2
                self.bridge_state["v2_type"] = "generated"
                self.bridge_state["temp_files"].append(new_v2)
                
                # 4. Stitch again
                self._stitch_preview()
            else:
                print("Regeneration failed.")
                self.root.after(0, self._enable_verif_ui) # Re-enable to try again
        else:
            print("Bridge frame extraction failed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VeoGUIApp(root)
    root.mainloop()
