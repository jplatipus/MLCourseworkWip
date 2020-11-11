clear
clc
clf
close all
load letterDatasetClass.mat

nBayes = NBayesClass(letterDatasetNormalised)
nBayes.simpleNaiveBayesClassifier()
