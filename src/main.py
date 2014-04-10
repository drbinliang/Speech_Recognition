'''
Created on 10/04/2014

@author: Bin Liang
'''
import os
from utils import Speech, SpeechRecognizer

CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # 10 categories


def loadData(dirName):
    ''' load data and do feature extraction '''
    fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']
    
    speechList = []
    
    for fileName in fileList:
        speech = Speech(dirName, fileName)
        speech.extractFeature()
        
        speechList.append(speech)
        
    return speechList


def training(speechList):
    ''' HMM training '''
    speechRecognizerList = []
    
    # initialize speechRecognizer
    for categoryId in CATEGORY:
        speechRecognizer = SpeechRecognizer(categoryId)
        speechRecognizerList.append(speechRecognizer)
    
    # organize data into the same category
    for speechRecognizer in speechRecognizerList:
        for speech in speechList:
            if speech.categoryId ==  speechRecognizer.categoryId:
                speechRecognizer.trainData.append(speech.features)
        
        # get hmm model
        speechRecognizer.getHmmModel()
    
    return speechRecognizerList
    

def recognize(testSpeechList, speechRecognizerList):
    ''' recognition ''' 
    predictCategoryIdList = []
    
    for testSpeech in testSpeechList:
        scores = []
        
        for recognizer in speechRecognizerList:
            score = recognizer.hmmModel.score(testSpeech.features)
            scores.append(score)
        
        idx = scores.index(max(scores))
        predictCategoryId = speechRecognizerList[idx].categoryId
        predictCategoryIdList.append(predictCategoryId)

    return predictCategoryIdList


def calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList):
    ''' calculate recognition rate '''
    score = 0
    length = len(groundTruthCategoryIdList)
    
    for i in range(length):
        gt = groundTruthCategoryIdList[i]
        pr = predictCategoryIdList[i]
        
        if gt == pr:
            score += 1
    
    recognitionRate = float(score) / length
    return recognitionRate
    

def main():
    
    ### Step.1 Loading training data
    print 'Step.1 Training data loading...',
    trainDir = './training_data/'
    trainSpeechList = loadData(trainDir)
    print 'done!'
    print
    
    ### Step.2 Training
    print 'Step.2 Training model...',
    speechRecognizerList = training(trainSpeechList)
    print 'done!'
    print
    
    ### Step.3 Loading test data
    print 'Step.3 Test data loading...',
    testDir = './test_data/'
    testSpeechList = loadData(testDir)
    print 'done!'
    print
    
    ### Step.4 Recognition
    print 'Step.4 Recognizing...'
    predictCategoryIdList = recognize(testSpeechList, speechRecognizerList)
    print
    
    ### Step.5 Print result
    groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    recognitionRate = calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList)
    
    print '===== Final result ====='
    print 'Ground Truth:\t', groundTruthCategoryIdList
    print 'Prediction:\t', predictCategoryIdList
    print 'Accuracy:\t', recognitionRate
    

if __name__ == '__main__':
    main()