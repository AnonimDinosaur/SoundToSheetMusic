# 🎶 Frequency-to-Note Converter  

## 📌 Introduction  
This is a **basic** real-time frequency-to-note converter. The program listens to an audio signal, extracts its dominant frequency, and maps it to the closest musical note. The project is still **in development** and lacks optimization, but it demonstrates the core concept of pitch detection.  

## ⚙️ How It Works  
1. The program records audio input from the microphone.  
2. It processes the signal to extract the dominant frequency.  
3. The detected frequency is converted into the corresponding musical note.  
4. The results are displayed on a simple Tkinter-based interface.

![Captura de pantalla 2025-03-01 214815](https://github.com/user-attachments/assets/fd5225a3-0f24-42de-ba09-7ff37b9b260c)

## 📂 Dependencies  
Make sure you have the following Python libraries installed:  
- `tkinter` (built-in)  
- `music21`  
- `numpy`  
- `matplotlib`  
- `os`  
- `librosa`  
- `pydub`  
- `soundfile`  
- `glob`  

You can install the required dependencies using:  
```bash
pip install music21 numpy matplotlib librosa pydub soundfile
```
## 🚀 Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/frequency-to-note.git
``` 
2. Install dependencies (if not already installed).
3. Run the script

## 🚧 Limitations
Basic implementation: The algorithm is not highly optimized.
Environmental noise: Background noise may affect accuracy.
Latency: Real-time detection may not be perfectly smooth.
## 🔧 Future Improvements
Better frequency detection algorithms
Noise filtering
Graphical improvements
## 📜 License
This project is licensed under the MIT License.
