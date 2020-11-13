clear
clc
clf
close all
load letterDatasetClass.mat

nBayes = NBayesClass(letterDatasetNormalised)
model = nBayes.simpleNaiveBayesClassifier()
