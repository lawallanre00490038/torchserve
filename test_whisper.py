import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
 
# Load the processor and model from the saved directories
processor = WhisperProcessor.from_pretrained("./whisper-large-processor")
model = WhisperForConditionalGeneration.from_pretrained("./whisper-large-model")
 
# Load an audio file using torchaudio (assuming you have a file named 'conversation.wav')
audio_input, sample_rate = torchaudio.load('./test_audio_files/conversation.wav')
 
# Resample the audio if necessary (Whisper models expect 16000 Hz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)
 
# Preprocess the audio input
input_features = processor(audio_input.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
 
# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
 
# Print the transcription
print(transcription[0])