import numpy as np
import sounddevice as sd
from scipy.signal import resample
from transformers import pipeline
import torch

class LiveTranscriber:
    def __init__(self, model_name="Tarakeshwaran/faster-whisper-small-en", device=None):
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
            task="automatic-speech-recognition",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        # Audio stream configurations
        self.target_sample_rate = 16000  # Whisper models use 16kHz
        self.chunk_duration = 5  # Seconds of audio to transcribe at once
        self.chunk_samples = self.target_sample_rate * self.chunk_duration
        
        # Get system default sample rate
        self.system_sample_rate = sd.query_devices(kind='input')['default_samplerate']
        print(f"Using device: {self.device}")
        print(f"System Sample Rate: {self.system_sample_rate}")
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for processing audio stream.
        
        :param indata: Input audio data
        :param frames: Number of frames
        :param time: Timestamp
        :param status: Stream status
        """
        if status:
            print(f"Stream error: {status}")
            return
        
        # Convert to numpy array and ensure mono
        audio = indata.flatten()
        
        # Resample to 16kHz if needed
        if self.system_sample_rate != self.target_sample_rate:
            audio = self._resample(audio, self.system_sample_rate, self.target_sample_rate)
        
        # Run transcription
        transcription = self.transcription_pipeline(audio)
        print(f"Transcription: {transcription['text']}")
    
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
            with sd.InputStream(
                samplerate=self.system_sample_rate,
                channels=1,  # Mono
                dtype='float32',
                callback=self.audio_callback,
                blocksize=int(self.system_sample_rate * self.chunk_duration)
            ):
                # Keep the main thread running
                while True:
                    sd.sleep(1000)
        
        except KeyboardInterrupt:
            print("\nTranscription stopped by user.")

def main():
    transcriber = LiveTranscriber()
    print("program start - wait till setup complete")
    transcriber.start_transcription()

if __name__ == "__main__":
    main()
