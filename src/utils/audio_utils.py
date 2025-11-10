import numpy as np
import librosa

#extract feature from the given audio
def extract_feature(file_path: str, n_mfcc: int = 40, max_pad_len = 174):
    """Extracts MFCC features from a given audio file.
    Pads or trims to a fixed length to ensure consistent CNN input shape

    Args:
        file_path (str): Path of the audio file (.wav expected)
        n_mfcc (int, optional): Number of MFCC coffiecients to extract. Defaults to 40.
        max_pad_len (int, optional): Maximum length (frames) for padding or trimming. Defaults to 174.

    Returns:
        np.ndarray: MFCC feature array of shape (n_mfcc, max_pad_len)
    """
    try:	
        audio, sr = librosa.load(file_path, res_type = 'kaiser_fast')
        mfccs = librosa.feature.mfcc(audio, sr = sr, n_mfcc =n_mfcc)

        # pad or truncate to fixed length
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width =((0,0), (0,pad_width)), mode = 'constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        
        return mfccs
    
    except Exception as e:
        print(f"Error happened while parsing the {file_path} : {e}")
        return None