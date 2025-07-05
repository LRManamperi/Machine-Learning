from speechbrain.inference.speaker import SpeakerRecognition
import torchaudio

# Load model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Use your own or sample files (must be 16kHz mono .wav)
score, prediction = verification.verify_files(
    "spk1_snt1.wav",  # Same speaker
    "spk1_snt2.wav"
)

print(f"Verification Score: {score}")
print("Prediction:", "Same speaker" if prediction == 1 else "Different speaker")

####
from speechbrain.inference.speaker import SpeakerRecognition

# Load model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Verify two audio files
score, prediction = verification.verify_files(
    "spk1_snt1.wav",  # First audio file
    "spk2_snt4.wav"   # Second audio file (different speaker)
)

print(f"Verification Score: {score}")
if prediction == 1:
    print("Prediction: Same speaker")
else:
    print("Prediction: Different speaker")
