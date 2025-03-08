import librosa
import numpy as np

def analyze_audio(audio_path, start_timestamp=0):
    """
    Analyze audio for suspicious sounds like background voices, phone notifications, etc.
    
    Args:
        audio_path: Path to the audio file
        start_timestamp: Starting timestamp for audio chunk analysis
        
    Returns:
        List of detected violations with description and confidence
    """
    violations = []
    
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract audio features
        # Spectral centroid - brightness of sound
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral bandwidth - range of frequencies
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Spectral contrast - difference between peaks and valleys in spectrum
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
        
        # MFCC - for voice characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Check for multiple voices using spectral contrast
        # Higher variance in spectral contrast can indicate multiple voices
        if np.std(spec_contrast) > 15:  # Threshold determined empirically
            violations.append({
                "description": "Multiple voices detected in audio",
                "confidence": 0.75,
                "timestamp": start_timestamp + np.argmax(spec_contrast) / sr
            })
        
        # Check for sudden noises (potential phone notifications, etc.)
        # Detect peaks in the audio amplitude
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        
        if len(peaks) > 5:  # Threshold for suspicious number of peaks
            violations.append({
                "description": "Unusual number of audio peaks detected (possible notifications)",
                "confidence": 0.65,
                "timestamp": start_timestamp + peaks[0] * librosa.get_duration(y=y, sr=sr) / len(onset_env)
            })
        
        # Check for consistent background noise
        percentile_5 = np.percentile(np.abs(y), 5)
        if percentile_5 > 0.05:  # Threshold for background noise
            violations.append({
                "description": "High level of consistent background noise detected",
                "confidence": 0.70,
                "timestamp": start_timestamp
            })
        
    except Exception as e:
        # Handle exceptions
        print(f"Error in audio analysis: {str(e)}")
    
    return violations
