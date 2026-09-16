import speech_recognition as sr
import time

def check_mic_levels():
    r = sr.Recognizer()
    
    # List mics again just to be sure
    print("\n--- Available Microphones ---")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{index}: {name}")
    print("-----------------------------\n")

    print("Using default microphone for test...")
    try:
        with sr.Microphone() as source:
            print("1. CALIBRATION TEST")
            print("Please remain SILENT for 2 seconds...")
            r.adjust_for_ambient_noise(source, duration=2)
            print(f"-> Calculated Energy Threshold (Silence): {r.energy_threshold}")
            print("(Lower is more sensitive. Usually 50-300 is good for quiet rooms. >1000 means noisy or bad mic)")
            
            print("\n2. SPEAKING TEST")
            print("Please SPEAK normally now (say 'Uno, Dos, Tres')...")
            print("Listening for 5 seconds...")
            
            # Record audio to check if it captures anything
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                print("-> Audio captured successfully.")
                
                # Try raw energy check (heuristic)
                # Not easily accessible from audio object directly without processing, 
                # but we can try recognition to see if it heard.
                try:
                    print("-> Attempting recognition...")
                    text = r.recognize_google(audio, language="es-ES")
                    print(f"-> RECOGNIZED: '{text}'")
                except sr.UnknownValueError:
                    print("-> Audio captured but NOT recognized (maybe too quiet or unintelligible).")
                except Exception as e:
                    print(f"-> Recognition API Error: {e}")
                    
            except sr.WaitTimeoutError:
                print("-> TIMEOUT: No speech detected over threshold.")
                
    except Exception as e:
        print(f"Error accessing microphone: {e}")

if __name__ == "__main__":
    check_mic_levels()