import tkinter as tk
from tkinter import filedialog, scrolledtext
from music21 import stream, note
from music21 import environment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import librosa
from pydub import AudioSegment
from pydub.utils import which
import warnings
import soundfile as sf
import glob


IMAGE_SIZE = 1200  # ↔ Amplada de la imatge en píxels (ajusta aquest valor)
IMAGE_DPI = 200  # 📏 Resolució de la imatge (p.ex. 100, 200, 300 dpi)

us = environment.UserSettings()
us['lilypondPath'] = r"C:\Users\oriol\OneDrive\Escritorio\Projects\lilypond-2.24.4\bin\lilypond.exe"

ffmpeg_path = r"C:\Users\oriol\OneDrive\Escritorio\escritori actual\ffmpeg-2025-02-26-git-99e2af4e78-full_build\bin"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")



class AudioAnalyzer:
    # Define note frequency ranges (Hz)
    NOTE_RANGES = {
    'C1':  (30, 33),  'C#1': (33, 35),  'D1':  (35, 38),  'D#1': (38, 41),
    'E1':  (41, 44),  'F1':  (44, 47),  'F#1': (47, 50),  'G1':  (50, 53),
    'G#1': (53, 56),  'A1':  (56, 60),  'A#1': (60, 64),  'B1':  (64, 68),

    'C2':  (68, 72),  'C#2': (72, 76),  'D2':  (76, 81),  'D#2': (81, 86),
    'E2':  (86, 91),  'F2':  (91, 96),  'F#2': (96, 102), 'G2':  (102, 108),
    'G#2': (108, 114), 'A2': (114, 121), 'A#2': (121, 128), 'B2': (128, 136),

    'C3':  (136, 144), 'C#3': (144, 152), 'D3': (152, 161), 'D#3': (161, 171),
    'E3':  (171, 181), 'F3':  (181, 192), 'F#3': (192, 203), 'G3':  (203, 215),
    'G#3': (215, 228), 'A3': (228, 242), 'A#3': (242, 256), 'B3': (256, 271),

    'C4':  (271, 287), 'C#4': (287, 303), 'D4': (303, 320), 'D#4': (320, 338),
    'E4':  (338, 357), 'F4':  (357, 376), 'F#4': (376, 397), 'G4':  (397, 419),
    'G#4': (419, 441), 'A4': (441, 466), 'A#4': (466, 491), 'B4': (491, 518),

    'C5':  (518, 545), 'C#5': (545, 573), 'D5': (573, 603), 'D#5': (603, 634),
    'E5':  (634, 666), 'F5':  (666, 700), 'F#5': (700, 735), 'G5':  (735, 771),
    'G#5': (771, 808), 'A5': (808, 847), 'A#5': (847, 897), 'B5': (897, 948),

    'C6':  (948, 1001), 'C#6': (1001, 1056), 'D6': (1056, 1112), 'D#6': (1112, 1170),
    'E6':  (1170, 1229), 'F6':  (1229, 1290), 'F#6': (1290, 1352), 'G6':  (1352, 1416),
    'G#6': (1416, 1482), 'A6': (1482, 1550), 'A#6': (1550, 1619), 'B6': (1619, 1690),

    'C7':  (1690, 1763), 'C#7': (1763, 1837), 'D7': (1837, 1913), 'D#7': (1913, 1991),
    'E7':  (1991, 2071), 'F7':  (2071, 2153), 'F#7': (2153, 2237), 'G7':  (2237, 2322),
    'G#7': (2322, 2410), 'A7': (2410, 2500), 'A#7': (2500, 2591), 'B7': (2591, 2684),

    'C8':  (2684, 2780)
    }
    
    def __init__(self):
        self.audio_path = None
        self.y = None
        self.sr = None
        self.results = []
        
    def load_audio(self, file_path):
        """Load audio file and convert to WAV if needed."""
        try:
            self.audio_path = file_path
            
            # Handle MP3 files by converting to WAV temporarily
            if file_path.lower().endswith('.mp3'):
                try:
                    audio = AudioSegment.from_mp3(file_path)
                    temp_wav = os.path.join(os.path.dirname(file_path), "temp_conversion.wav")
                    audio.export(temp_wav, format="wav")
                    
                    # Use soundfile instead of deprecated librosa loading
                    data, samplerate = sf.read(temp_wav)
                    self.y = data
                    if len(data.shape) > 1:  # Convert stereo to mono by averaging channels
                        self.y = np.mean(data, axis=1)
                    self.sr = samplerate
                    
                    # Clean up temp file
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
                except Exception as e:
                    print(f"Error with MP3 conversion: {e}")
                    return False
            else:
                # Direct WAV loading with soundfile
                try:
                    data, samplerate = sf.read(file_path)
                    self.y = data
                    if len(data.shape) > 1:  # Convert stereo to mono by averaging channels
                        self.y = np.mean(data, axis=1)
                    self.sr = samplerate
                except Exception as e:
                    print(f"Error with WAV loading via soundfile: {e}")
                    # Fallback to librosa with warning suppression
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.y, self.sr = librosa.load(file_path, sr=None, mono=True)
            
            # Ensure data is loaded
            if self.y is None or self.sr is None:
                print("Failed to load audio data")
                return False
                
            print(f"Successfully loaded audio: {len(self.y)} samples at {self.sr}Hz")
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def get_note_from_frequency(self, freq):
        
                if freq < 20 or freq > 4000:  # Fora del rang del piano
                    return None

                for note, (low, high) in self.NOTE_RANGES.items():
       
                    if low <= freq < high:
        
                        return note
                return None

    
    def analyze_audio(self, silence_threshold=-60):
        """Analyze audio to detect notes and silences over time."""
        try:
            if self.y is None or self.sr is None:
                print("No audio data loaded")
                return False
                
            # Calculate frame length for chunking the audio (e.g., 100ms frames)
            frame_length = int(self.sr * 0.1)
            hop_length = frame_length // 2
            
            # Ensure frame length is even to avoid librosa errors
            if frame_length % 2 != 0:
                frame_length += 1
            
            print(f"Analyzing audio with frame length: {frame_length}, hop length: {hop_length}")
            
            # Get dominant frequencies over time
            try:
                S = librosa.stft(self.y, n_fft=2048, hop_length=hop_length)
                D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            except Exception as e:
                print(f"Error in STFT calculation: {e}")
                return False
            
            print(f"STFT completed. Shape: {S.shape}")
            
            # Extract timing and dominant frequencies
            times = librosa.times_like(S, sr=self.sr, hop_length=hop_length)
            
            # Find the dominant frequency at each time frame
            results = []
            current_note = None
            current_start = 0
            
            print(f"Processing {len(times)} time frames")
            
            for i, time in enumerate(times):
                if i >= len(D[0]):
                    print(f"Reached end of frequency data at frame {i}")
                    break
                    
                # Get amplitude for this frame
                frame_amplitude = np.max(D[:, i])
                
                # Check if this is silence
                if frame_amplitude < silence_threshold:
                    # If we were tracking a note, end it
                    if current_note is not None and current_note != 'Silence':
                        duration = time - current_start
                        if duration > 0.05:  # Minimum duration threshold (50ms)
                            results.append((current_note, current_start, duration))
                        current_note = None
                    
                    # If we're starting silence or continuing silence
                    if current_note != 'Silence':
                        current_start = time
                        current_note = 'Silence'
                else:
                    # Find the frequency bin with maximum energy
                    freq_bin = np.argmax(D[:, i])
                    
                    # Check if the bin index is valid
                    if freq_bin < len(librosa.fft_frequencies(sr=self.sr, n_fft=2048)):
                        freq = librosa.fft_frequencies(sr=self.sr, n_fft=2048)[freq_bin]
                        
                        # Map to a note
                        note = self.get_note_from_frequency(freq)
                        
                        # Handle note transitions or the start of a note
                        if note != current_note:
                            # End the previous note/silence if there was one
                            if current_note is not None:
                                duration = time - current_start
                                if duration > 0.05:  # Minimum duration threshold (50ms)
                                    results.append((current_note, current_start, duration))
                            
                            # Start the new note
                            current_start = time
                            current_note = note
            
            # Add the final note/silence
            if current_note is not None and len(times) > 0:
                duration = times[-1] - current_start
                if duration > 0.05:  # Minimum duration threshold (50ms)
                    results.append((current_note, current_start, duration))
            
            print(f"Analysis complete. Found {len(results)} note/silence segments")
            self.results = results
            return True
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return False
    
    def get_results_text(self):
        """Format results into a readable string."""
        if not self.results:
            return "No results available. Please analyze an audio file first."
            
        text = ""
        for note, start, duration in self.results:
            if note == 'Silence':
                text += f"Silence - {duration:.1f}s "
            elif note is None:
                text += f"Unknown - {duration:.1f}s "
            else:
                text += f"{note} - {duration:.1f}s "
                
        return text.strip()
    
    def create_visualization(self, figure):
        """Create visualization of the audio analysis."""
        try:
            if self.y is None:
                print("No audio data for visualization")
                return False
                
            if not self.results:
                print("No analysis results for visualization")
                return False
                
            # Clear the figure
            figure.clear()
            ax = figure.add_subplot(111)
            
            # Create a waveform plot
            ax.plot(np.linspace(0, len(self.y)/self.sr, len(self.y)), self.y)
            
            # Highlight different notes with colors
            unique_notes = set(note for note, _, _ in self.results if note != 'Silence' and note is not None)
            
            if unique_notes:  # Check if there are any notes to avoid empty set
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_notes)))
                color_map = {}
                color_idx = 0
                
                for note, start, duration in self.results:
                    if note == 'Silence':
                        color = 'gray'
                    elif note is None:
                        color = 'lightgray'
                    else:
                        if note not in color_map:
                            color_map[note] = colors[color_idx]
                            color_idx += 1
                        color = color_map[note]
                    
                    # Add colored span
                    ax.axvspan(start, start + duration, alpha=0.3, color=color)
                    
                    # Add text label if span is wide enough to display text
                    if duration > 0.2:  # Only add text for segments longer than 200ms
                        if note == 'Silence':
                            label = "Silent"
                        elif note is None:
                            label = "?"
                        else:
                            label = note
                        ax.text(start + duration/2, 0, label, horizontalalignment='center')
            
            # Set axis labels and title
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Audio Analysis')
            
            figure.tight_layout()
            return True
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return False

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio to Musical Notes Analyzer")
        self.root.geometry("800x600")
        
        self.analyzer = AudioAnalyzer()
        
        # Create frames
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        self.import_btn = tk.Button(self.control_frame, text="Import Audio", command=self.import_audio)
        self.import_btn.pack(side=tk.LEFT, padx=5)
        
        # Fix: Initialize transform button as disabled until file is loaded
        self.transform_btn = tk.Button(self.control_frame, text="Transform", command=self.transform_audio)
        self.transform_btn.pack(side=tk.LEFT, padx=5)
        self.transform_btn.config(state=tk.DISABLED)
        
        self.file_label = tk.Label(self.control_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        # Status label for feedback
        self.status_label = tk.Label(self.control_frame, text="", fg="blue")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Results area
        self.result_label = tk.Label(self.result_frame, text="Results:")
        self.result_label.pack(anchor=tk.W)
        
        self.result_text = scrolledtext.ScrolledText(self.result_frame, height=4)
        self.result_text.pack(fill=tk.X, pady=5)
        
        # Visualization area
        self.viz_label = tk.Label(self.result_frame, text="Visualization:")
        self.viz_label.pack(anchor=tk.W, pady=(10, 0))
        
        self.figure = plt.Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def import_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3")]
        )
        
        if file_path:
            self.status_label.config(text="Loading file...", fg="blue")
            self.root.update()  # Update the UI to show loading message
            
            self.file_label.config(text=os.path.basename(file_path))
            success = self.analyzer.load_audio(file_path)
            
            if success:
                self.transform_btn.config(state=tk.NORMAL)  # Enable transform button
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "File loaded successfully. Click 'Transform' to analyze.")
                self.figure.clear()
                self.canvas.draw()
                self.status_label.config(text="File loaded successfully", fg="green")
            else:
                self.transform_btn.config(state=tk.DISABLED)  # Keep transform button disabled
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Error loading audio file. Please try another file.")
                self.status_label.config(text="Error loading file", fg="red")
        else:
            # User cancelled file selection
            self.status_label.config(text="File selection cancelled", fg="blue")
                
    def transform_audio(self):
        if not hasattr(self.analyzer, 'y') or self.analyzer.y is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "No audio file loaded. Please import a file first.")
            self.status_label.config(text="No file loaded", fg="red")
            return
            
        self.status_label.config(text="Analyzing audio...", fg="blue")
        self.root.update()  # Update the UI to show analyzing message
        
        success = self.analyzer.analyze_audio()
        
        if success:
            results_text = self.analyzer.get_results_text()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, results_text)
            
            # Create visualization
            viz_success = self.analyzer.create_visualization(self.figure)
            if viz_success:
                self.canvas.draw()
                self.status_label.config(text="Analysis complete", fg="green")
            else:
                self.status_label.config(text="Analysis complete, but visualization failed", fg="orange")
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Error analyzing audio. Please try another file.")
            self.status_label.config(text="Analysis failed", fg="red")

def generate_sheet_music(note_sequence):
    """Converteix la seqüència de notes en una partitura Music21."""
    score = stream.Stream()
    notes = note_sequence.split(" - ")
    
    for item in notes:
        parts = item.split("s ")
        if parts[0] == "Silence":
            duration = 0.5  # Duració per defecte per silencis
        else:
            duration = float(parts[0].replace("s", ""))  # Elimina la "s" i converteix a float

        if len(parts) > 1:
            pitch = parts[1]
            if pitch != "Silence":
                n = note.Note(pitch)
                n.quarterLength = duration * 4  # Convertir segons a duració musical
                score.append(n)
            else:
                rest = note.Rest()
                rest.quarterLength = duration * 4
                score.append(rest)
    
    return score


def draw_sheet_music(score, frame):
    """Dibuixa la partitura en la UI amb Matplotlib."""
    fig, ax = plt.subplots(figsize=(60, 40))
    
    score_lilypond = score.write(fmt='lily.png')  # Genera la partitura com a imatge
    img = plt.imread(score_lilypond)
    ax.imshow(img)
    ax.axis('off')
    
    for widget in frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def generate_lilypond(score):
    lilypond_code = score.write(fmt='lilypond')
    
    # Llegeix el fitxer LilyPond generat
    with open(lilypond_code, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Afegeix la configuració després de \version
    for i, line in enumerate(lines):
        if "\\version" in line:
            lines.insert(i + 1, """
\paper {
  system-system-spacing.basic-distance = #12
  page-breaking = #ly:optimal-breaking
}
""")
            break

    # Guarda el fitxer modificat
    with open(lilypond_code, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return lilypond_code

def main():
    root = tk.Tk()
    root.title("Detecció de Notes i Generació de Partitura")
    
    app = AudioAnalyzerApp(root)
    frame_sheet = tk.Frame(root)
    frame_sheet.pack(pady=1)
    
    def generate_and_show_sheet():
        """Aconsegueix les notes detectades i les mostra com a partitura."""
        note_sequence = app.analyzer.get_results_text()
        score = generate_sheet_music(note_sequence)
        draw_sheet_music(score, frame_sheet)
    
    sheet_button = tk.Button(root, text="Generar Partitura", command=generate_and_show_sheet)
    sheet_button.pack()
    
    root.mainloop()


if __name__ == "__main__":
    main()
