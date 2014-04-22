'''
Created on 10/04/2014

@author: Bin Liang
'''
from scikits.talkbox.features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm

class Speech:
    ''' speech class '''
    
    def __init__(self, dirName, fileName):
        self.fileName = fileName    # file name
        self.features = None    # feature matrix
        self.soundSamplerate, self.sound = wavfile.read(dirName + fileName)
        
        # category assignment
        idx1 = self.fileName.find('_')
        idx2 = self.fileName.find('.')
        self.categoryId = fileName[idx1 + 1 : idx2]   # speech category
        
        
    def extractFeature(self):
        ''' mfcc feature extraction '''
        self.features = mfcc(self.sound, nwin=int(self.soundSamplerate * 0.03), fs=self.soundSamplerate, nceps=13)[0]

class SpeechRecognizer:
    ''' class for speech recognizer '''
    
    def __init__(self, categoryId):
        self.categoryId = categoryId
        self.trainData = []
        self.hmmModel = None
        
    def getHmmModel(self):
        ''' get hmm model from training data '''
        
        numStates = 10
        model = hmm.GaussianHMM(numStates, "diag") # initialize hmm model
        model.fit(self.trainData)   # get optimal parameters
        self.hmmModel = model
        
    
