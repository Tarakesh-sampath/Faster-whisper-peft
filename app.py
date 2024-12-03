import numpy as np
import pyaudio
import wave
from scipy.signal import resample
from transformers import pipeline
import torch
import warnings
import queue
import threading
warnings.filterwarnings("ignore")

class LiveTranscriber:
    def __init__(self, model_name="openai/whisper-small.en", device=None):
        """
        Initialize the live transcription system.
        
        :param model_name: Hugging Face model for transcription
        :param device: Compute device (cuda/cpu)
        """
        # Determine optimal device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load Whisper transcription pipeline
        self.transcription_pipeline = pipeline(
            task='automatic-speech-recognition',
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        # Audio stream configurations
        self.target_sample_rate = 16000  # Whisper models use 16kHz
        self.chunk_duration = 4  # Seconds of audio to transcribe at once
        self.chunk_samples = self.target_sample_rate * self.chunk_duration
        
        # PyAudio setup
        self.pyaudio_instance = pyaudio.PyAudio()
        
        # Get system default input device
        self.system_sample_rate = int(self.pyaudio_instance.get_default_input_device_info()['defaultSampleRate'])
        
        # Audio queue for thread-safe processing
        self.audio_queue = queue.Queue()
        
        # Transcription thread
        self.transcription_thread = None
        self.is_running = False
        
        print(f"Using device: {self.device}")
        print(f"System Sample Rate: {self.system_sample_rate}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for PyAudio stream.
        
        :param in_data: Input audio data
        :param frame_count: Number of frames
        :param time_info: Time information
        :param status: Stream status
        """
        # Convert input data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def _transcription_worker(self):
        """
        Background worker for processing audio chunks and transcribing
        """
        while self.is_running:
            try:
                # Get audio chunk with timeout to allow checking is_running
                audio = self.audio_queue.get(timeout=1)
                
                # Resample to 16kHz if needed
                if self.system_sample_rate != self.target_sample_rate:
                    audio = self._resample(audio, self.system_sample_rate, self.target_sample_rate)
                
                # Run transcription
                transcription = self.transcription_pipeline(audio)
                print(f"Transcription: {transcription['text']}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
    
    def _resample(self, audio, orig_sr, target_sr):
        """
        Resample audio to target sample rate.
        
        :param audio: Input audio array
        :param orig_sr: Original sample rate
        :param target_sr: Target sample rate
        :return: Resampled audio
        """
        # Calculate number of samples for resampling
        duration = len(audio) / orig_sr
        num_samples = int(duration * target_sr)
        
        # Use scipy's resample function
        return resample(audio, num_samples)
    
    def start_transcription(self):
        """
        Start live audio transcription with keyboard interrupt support.
        """
        print(f"Starting transcription on {self.device}. Speak now...")
        print("Press Ctrl+C to stop transcription.")
        
        try:
            # Open audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,  # Mono
                rate=self.system_sample_rate,
                input=True,
                frames_per_buffer=int(self.system_sample_rate * self.chunk_duration),
                stream_callback=self._audio_callback
            )
            
            # Start transcription thread
            self.is_running = True
            self.transcription_thread = threading.Thread(target=self._transcription_worker)
            self.transcription_thread.start()
            
            # Keep main thread running
            self.transcription_thread.join()
        
        except KeyboardInterrupt:
            print("\nTranscription stopped by user.")
        finally:
            # Cleanup
            self.is_running = False
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            self.pyaudio_instance.terminate()
            if self.transcription_thread:
                self.transcription_thread.join()

def main():
    print("program start - wait till setup complete")
    transcriber = LiveTranscriber()
    transcriber.start_transcription()

if __name__ == "__main__":
    main()