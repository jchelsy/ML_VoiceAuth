# Import Libs
import os
import pickle
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
from src import cwd
# Ignore warnings
warnings.filterwarnings("ignore")


# Method for calculating & returning the delta of a given feature vector matrix
def calculate_delta(array):
    # Get the data shape (rows / columns)
    rows, cols = array.shape
    # Initialize a new array
    deltas = np.zeros((rows, 20))

    # Calculate the delta feature from MFCC vector (n = number of deltas summed over)
    n = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= n:
            if i - j < 0:
                first = 0
            else:
                first = i - j

            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j

            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10

    return deltas


# Method for extracting audio features
def extract_features(audio, rate):
    # Extract 20-dim MFCC features from audio
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    # Normalize MFCC features
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # Combine the delta (to convert the MFCC vector to a 40-dim feature vector matrix)
    delta = calculate_delta(mfcc_feature)
    # Stack the feature vectors into a combined array
    combined = np.hstack((mfcc_feature, delta))
    # Return the result features
    return combined


# Method for training the GMM models
def train_model():
    # Initialize the necessary directory paths
    source_path = os.path.join(cwd, "data", "training_set")
    model_path = os.path.join(cwd, "data", "trained_models")

    # Initialize a counter for tracking the number of files that have been analyzed
    #   (Every iteration of 5 training files is an entire set - more than 5 files = 2+ training sets!!!)
    count = 1
    # Initialize a numpy array for the training features
    features = np.asarray(())

    # Iterate through all files in the training set directory
    for file in os.listdir(source_path):
        print(file)  # Print filenames

        # Read the audio file
        sr, audio = read(os.path.join(source_path, file))
        # Extract the audio wave vector features from the audio
        vector = extract_features(audio, sr)

        # If this is the FIRST file in the training set
        if features.size == 0:
            # SET the features array to the vector of extracted features
            features = vector
        # Otherwise (if there are already vector(s) in the features array)
        else:
            # ADD the new features vector to the array!
            features = np.vstack((features, vector))

        # If the count is 5 (there are 5 training files per set, when count is 5, all of 1 audio set is complete)
        if count == 5:
            ######################
            # Training the Model #
            ######################

            # Initialize the Gaussian mixture model (for computing the audio vectors)
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            # Fit the Gaussian mixture model to create the training model
            gmm.fit(features)

            ######################################
            # Dumping the Trained Gaussian Model #
            ######################################

            # Retrieve the filename for the model (from the name in the training sample files, before the '-')
            pickle_filename = file.split("-")[0] + ".gmm"

            # Dump the trained model to a file
            pickle.dump(gmm, open(os.path.join(model_path, pickle_filename), 'wb'))

            # Output the filename
            print("==>", pickle_filename)
            print()  # Output a newline

            # RESET the features (for the next training model, if there are more training files!)
            features = np.asarray(())

            # Set the count back to zero (if there are more files, the next 5 files will be for another training set!)
            count = 0

        # Add 1 to the count (this happens for each iteration of every training file)
        #   (This counter is for tracking all 5 training files for each training set!)
        count += 1


# Method for testing both of the trained models ("Jonah" and "Jill")
def test_models():
    # Initialize the necessary directory paths
    source_path = os.path.join(cwd, "data", "testing_set")
    model_path = os.path.join(cwd, "data", "trained_models")

    # Retrieve training model files
    gmm_files = [os.path.join(model_path, filename) for filename in os.listdir(model_path) if filename.endswith(".gmm")]

    # Load the trained model files
    models = [pickle.load(open(filename, 'rb')) for filename in gmm_files]
    # Retrieve a list of the names of each speaker (obtained from trained model filenames)
    speakers = [filename.split("\\")[-1].split(".gmm")[0] for filename in gmm_files]

    # Loop through each test file (to test every sample against the trained models)
    for test_filename in os.listdir(source_path):
        # Read the audio file
        sr, audio = read(os.path.join(source_path, test_filename))

        # Extract 40-dimensional MFCC & delta MFCC features
        vector = extract_features(audio, sr)

        # Initialize a new array for likelihood comparison of the training models
        log_likelihood = np.zeros(len(models))

        # Loop through the # of training models (counting 'i' as the index)
        for i in range(len(models)):
            # Retrieve the trained Gaussian Mixture model
            gmm = models[i]
            # Score the accuracy of the testing feature vector with the GMM model
            scores = np.array(gmm.score(vector))
            # Add the accuracy score to the 'log_likelihood' array
            log_likelihood[i] = scores.sum()

        # Retrieve the highest accuracy score (from the 'log_likelihood' array)
        result = np.argmax(log_likelihood)

        # Output the result to the console
        print(test_filename, "\n\tSpeaker: ", speakers[result])


if __name__ == "__main__":
    # train_model()  # Train the models for all training sets ( FORMAT:  "[Name]-sample0"  -  "[Name]-sample4" )
    test_models()  # Test the trained models for all sample audio files (in 'testing_set' directory)
