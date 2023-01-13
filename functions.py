import librosa
from librosa import power_to_db , util
import librosa.display
import IPython.display as ipd
# import csv
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report 
import scipy
import os
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import get_window
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
import glob
import pickle
import joblib
from sklearn.metrics import f1_score
from sklearn import preprocessing
import python_speech_features as mfcc
import matplotlib
import matplotlib.pyplot as plt

def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(file_path):
    # audio , sample_rate = librosa.load(file_path, mono=True, duration=2)
    # print(np.shape(audio))
    # y,index=librosa.effects.trim(audio,top_db=55)
    # audio = y[index[0]:index[1]]
    sr,audio = read(file_path)
    mfcc_feature = mfcc.mfcc(audio,sr, 0.025, 0.01,20, nfft = 1200 ,appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined     

def extractFromFile(directory):
    extractedFeatures = np.asarray(())
    for audio in os.listdir(directory):
        audio_path = directory + audio
        # print(audio_path)
        features   = extract_features(audio_path)
        if extractedFeatures.size == 0:
            extractedFeatures = features
        else:
            extractedFeatures = np.vstack((extractedFeatures, features))
    return extractedFeatures

def generateModel(modelName,features, pickleName):
    modelName = GaussianMixture(n_components=  6, max_iter = 200, covariance_type='spherical',n_init =3)
    modelName.fit(features)
    labels = modelName.predict(features)
    gmm = '.gmm'
    name = pickleName + gmm
    pickle.dump(modelName,open(name,'wb'))
    return modelName, labels

def plot_barChart(scores, speakerFlag, names, img):
    fig = plt.figure(figsize=(25,10))
    if speakerFlag:
        left = [1, 2, 3]
        height = np.add(scores,[100,100,100])
        plt.xlabel('Team members',fontsize=30)
        # plt.title('Speaker Recognition',fontsize=20)
        plt.axhline(y=max(height)-1,linewidth=4,color= 'coral', label = 'Threshold Score')
    else:
        left = [1,2,3,4]
        height = np.add(scores,[100,100,100,100])
        plt.xlabel('Words',fontsize=30)
    plt.bar(left, height, tick_label = names, width = 0.5, color = [ (0.7725490196078432, 0.6901960784313725, 0.8352941176470589)]) #, color = ['blue', 'cadetblue', 'cornflowerblue']
    plt.axhline(y=max(height), linewidth=4, color= 'purple', label = 'Maximum Score')
    
    plt.tick_params(axis='x', labelsize=25)
    plt.tick_params(axis='y', labelsize=25)
    # plt.axhline(y=max(height)-1,linewidth=4,color= 'r')
    plt.ylabel('Scores',fontsize=30)
    plt.legend(loc='lower right',fontsize=30)
    imagename = './static/'+ img + '.png'
    plt.savefig(imagename)

def plot_melspectrogram(file_name):
    audio,sfreq = librosa.load(file_name)
    # fig=plt.figure(figsize=(25,10))
    fig = plt.figure()
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sfreq)
    img=librosa.display.specshow(librosa.power_to_db(melspectrogram,ref = np.max))
    return img,fig

def spectral_Rolloff(file_name, img, min_percent):
    y , sr = librosa.load(file_name)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=min_percent)
    S, _ = librosa.magphase(librosa.stft(y=y))
    _, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
    ax.plot(librosa.times_like(rolloff), rolloff[0])
    ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w')
    
    plt.axhline(y=800, linewidth=2, color= 'w', label = 'Y1 = 800hz')
    plt.axhline(y=400, linewidth=2, color= 'w', label = 'Y1 = 400hz')
    ax.legend(loc='lower right') 
    ax.set(title='log Power spectrogram')
    imagename = './static/'+ img + '.png'
    plt.savefig(imagename)


# Works as a filter(low pass and hieght pass filter)
# ckeck if a center freq for a spectogram bins at least least has roll-of percentage (0.85) from power
def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True,  pad_mode="constant",
   freq=None, roll_percent=0.85 ):

    S, n_fft = librosa.core.spectrum._spectrogram( y=y,S=S,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window, center=center,
       pad_mode=pad_mode,)
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # reshaping
    if freq.ndim == 1:
       freq = util.expand_to(freq, ndim=S.ndim, axes=-2)
    # calculating total energy   
    total_energy = np.cumsum(S, axis=-2)
    # calculating the edges
    threshold = roll_percent * total_energy[..., -1, :]
    #reshaoing
    threshold = np.expand_dims(threshold, axis=-2)
    #if total energy of centeral freq < threshold (it is out of my edges
    # )
    ind = np.where(total_energy < threshold, np.nan, 1)
    spectral_rolloff=np.nanmin(ind * freq, axis=-2, keepdims=True)
    return spectral_rolloff


# def mfccPlotting(file_name, img):
#     # import librosa
#     # import matplotlib.pyplot as plt

#     COEFFICIENT_TO_ANALYZE = 2
#     N_BINS = 20

#     y, sr = librosa.load((file_name), sr=None)
#     mfcc_coeff = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

#     fig, ax = plt.subplots()
#     ax.hist(mfcc_coeff[COEFFICIENT_TO_ANALYZE, :], N_BINS)
#     ax.set_title(f'MFCC Coefficient {COEFFICIENT_TO_ANALYZE}')
#     imagename = './static/'+ img + '.png'
#     plt.savefig(imagename)
#     # plt.show()

# def extract_features_for_data(audio,sr):
#     # audio , sample_rate = librosa.load(file_path, mono=True ,duration=2 )
#     # print(np.shape(audio))
#     # sr,audio = read(file_path)
#     mfcc_feature = mfcc.mfcc(audio,sr, 0.025, 0.01,20, nfft = 1200 ,appendEnergy=True)
#     mfcc_feature = preprocessing.scale(mfcc_feature)
#     delta = calculate_delta(mfcc_feature)
#     combined = np.hstack((mfcc_feature,delta)) 
#     return combined

# def mfccc( y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0):
#     # db scale to colour code
#     S = power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
#     M = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]
#     if lifter > 0:
#         # reshaping
#         LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
#         LI = util.expand_to(LI, ndim=S.ndim, axes=-2)
#         #formula 
#         M *= 1 + (lifter / 2) * LI
#         return M
#     elif lifter == 0:
#         return M

# #===========================================================================
# #  when I will take the freq =0 (freq with low mag)
# def zero_crossing_rate(y,  frame_length=2048, hop_length=512, **kwargs):
#     y_framed = util.frame(y, frame_length=frame_length, hop_length=hop_length)
#     #reshaping
#     kwargs["axis"] = -2
#     # zero_crossing is a freq at which signal cross the axis 
#     crossings = librosa.zero_crossings(y_framed, **kwargs)
#     # mean of crossing
#     zero_crossing_rate=np.mean(crossings, axis=-2, keepdims=True)
#     return zero_crossing_rate

# #===========================================================================
# # represent center og mass of freq and using to predict brightness in the audio
# def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window="hann",center=True, pad_mode="constant" ):
#    magnitude, n_fft = librosa.core.spectrum._spectrogram( y=y,S=S,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window,
#        center=center, pad_mode=pad_mode)
#        # S is the spectogram Magnitude
#        #
#    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)   
#    if freq.ndim == 1:
#     #  (just reshaping)
#       freq = util.expand_to(freq, ndim=magnitude.ndim, axes=-2)   
#     # spectral centroif formela  (weighted freq (magnitude) * centeral freq)
#    spectral_centroid=np.sum(freq * util.normalize(magnitude, norm=1, axis=-2), axis=-2, keepdims=True)   
#    return spectral_centroid

# #===========================================================================
# # distance between min and max and min freq
# def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True,
#     pad_mode="constant", freq=None, centroid=None, norm=True, p=2 ):
#     S, n_fft = librosa.core.spectrum._spectrogram( y=y, S=S, n_fft=n_fft,hop_length=hop_length, win_length=win_length,window=window,center=center,
#        pad_mode=pad_mode)
       
#     centroid = spectral_centroid( y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq)
#     freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     if freq.ndim == 1:
#         deviation = np.abs(
#             np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1)
#         )
#     else:
#      deviation = np.abs(freq - centroid)
#     if norm:
#         # S is a weighted freq
#         S = util.normalize(S, norm=1, axis=-2)
#         # formela
#     spectral_bandwidth=np.sum(S * deviation ** p, axis=-2, keepdims=True) ** (1.0 / p)
#     return spectral_bandwidth

# #===========================================================================
# # root mean squar for each frame  samples and spectogram
# def rms(y=None, S=None, frame_length=2048, hop_length=512):
#     # samples 
#     if y is not None:
#         x = util.frame(y, frame_length=frame_length, hop_length=hop_length)
#         power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
#      # spectogram   
#     elif S is not None:
#         x = np.abs(S) ** 2
#         power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length ** 2
#     rms=np.sqrt(power)
#     return rms

# #===========================================================================

# def writeTocsv(data,csvName):
#     file = open(csvName, 'a', newline='')
#     writer = csv.writer(file)
#     writer.writerow(data.split(","))
#     file.close()



# def extract_features(directory, filename, csvName):
#     path = directory + filename

#     y,sr = librosa.load(path)
#     y, index = librosa.effects.trim(y)

#     rmse = rms(y=y)
#     spec_cent = spectral_centroid(y=y, sr=sr)
#     spec_bw = spectral_bandwidth(y=y, sr=sr)
#     rolloff = spectral_rolloff(y=y, sr=sr)
#     zcr = zero_crossing_rate(y)
#     mfcc = mfccc(y=y, sr=sr,n_mfcc=20)

#     # to_append = f'{filename},{np.mean(rmse)},{np.mean(spec_cent)},{np.mean(spec_bw)},{np.mean(rolloff)},{np.mean(zcr)}'
#     to_append = f'{np.mean(rmse)},{np.mean(spec_cent)},{np.mean(spec_bw)},{np.mean(rolloff)},{np.mean(zcr)}'
    
#     for e in mfcc:
#         to_append += f',{np.mean(e)}'

#     writeTocsv(to_append,csvName)

#     return to_append  

# def startCSV(csvName):
#     header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
#     for i in range(1, 21):
#         header += f' mfcc{i}'
#     # header += ' label'
#     header = header.split()
#     file = open(csvName, 'w', newline='')
#     writer = csv.writer(file)
#     writer.writerow(header)
#     file.close()


# def extract_features_array(filename):
#     feature = []

#     y,sr = librosa.load(filename)
#     y, index = librosa.effects.trim(y)

#     rmse = rms(y=y)
#     spec_cent = spectral_centroid(y=y, sr=sr)
#     spec_bw=spectral_bandwidth(y=y, sr=sr)
#     rolloff = spectral_rolloff(y=y, sr=sr)
#     zcr = zero_crossing_rate(y)
#     feature.append(np.mean(rmse))
#     feature.append(np.mean(spec_cent))
#     feature.append(np.mean(spec_bw))
#     feature.append(np.mean(rolloff))
#     feature.append(np.mean(zcr))

#     mfcc = mfccc(y=y, sr=sr,n_mfcc=20)
#     for e in mfcc:
#         feature.append(np.mean(e))
    
#     return feature  
