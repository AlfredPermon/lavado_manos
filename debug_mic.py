import speech_recognition as sr
try:
    print("Microphones found:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Microphone with name \"{name}\" found for `Microphone(device_index={index})`")
except Exception as e:
    print(f"Error listing microphones: {e}")
