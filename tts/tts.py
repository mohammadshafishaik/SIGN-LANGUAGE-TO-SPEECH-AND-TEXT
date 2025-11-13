import pyttsx3
import platform

class TextToSpeech:
    """
    A wrapper class for text-to-speech (TTS) conversion.
    It uses pyttsx3 for cross-platform support and can be configured
    for different languages and voices.
    """
    def __init__(self, rate=150, volume=0.9):
        """
        Initializes the TTS engine.

        Args:
            rate (int): The speaking rate (words per minute).
            volume (float): The volume of the speech (0.0 to 1.0).
        """
        self.engine = pyttsx3.init()
        self.system = platform.system()
        
        # Set properties
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Voice selection (can be customized)
        self.set_voice('english') # Default to English

    def list_voices(self):
        """Lists all available voices on the system."""
        voices = self.engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"Voice {i}:")
            print(f"  ID: {voice.id}")
            print(f"  Name: {voice.name}")
            print(f"  Lang: {voice.languages}")
            print(f"  Gender: {voice.gender}")
            print("-" * 20)

    def set_voice(self, language, gender='female'):
        """
        Sets the voice based on language and gender.
        Note: Voice availability is system-dependent.

        Args:
            language (str): The desired language (e.g., 'english', 'telugu').
            gender (str): The desired gender ('female' or 'male').
        """
        voices = self.engine.getProperty('voices')
        
        # Attempt to find a matching voice
        # This is a heuristic and may need to be adjusted based on your system's voice names
        for voice in voices:
            # Simple check for language name in voice name or language property
            lang_found = any(language in lang.lower() for lang in voice.languages)
            
            if lang_found and gender.lower() == voice.gender.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"Set voice to: {voice.name}")
                return
        
        print(f"Warning: Could not find a '{gender}' voice for '{language}'. Using default.")

    def speak(self, text, wait=False):
        """
        Converts the given text to speech.

        Args:
            text (str): The text to be spoken.
            wait (bool): If True, the function will block until speech is finished.
                         If False (default), it speaks asynchronously.
        """
        if not text:
            return
            
        self.engine.say(text)
        
        if wait:
            self.engine.runAndWait()
        else:
            # To make it non-blocking, we need to handle the event loop
            self.engine.startLoop(False)
            self.engine.iterate()
            self.engine.endLoop()


def main():
    """
    A simple demonstration of the TextToSpeech class.
    """
    print("Initializing Text-to-Speech engine...")
    tts = TextToSpeech()

    print("\nAvailable voices on this system:")
    tts.list_voices()

    print("\n--- English Demo ---")
    tts.set_voice('english', 'female')
    tts.speak("Hello, this is a test of the real-time sign language translator.", wait=True)
    
    # --- Telugu Demo ---
    # Note: A Telugu voice must be installed on your macOS/Windows system for this to work.
    # On macOS, go to System Settings > Accessibility > Spoken Content > System Voice,
    # and download a Telugu voice if available.
    print("\n--- Telugu Demo (requires a Telugu voice to be installed) ---")
    tts.set_voice('telugu', 'female')
    # The text must be in Telugu script for the engine to recognize it correctly.
    telugu_text = "నమస్కారం, ఇది ఒక పరీక్ష" # (Namaskaram, idi oka pariksha)
    tts.speak(telugu_text, wait=True)

    print("\nTTS Demo complete.")


if __name__ == "__main__":
    main()
