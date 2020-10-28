%
%% Test the dataset class:
% loads the dataset and checks the dimensions of the data are
% as expected
%
clear
clc
clf
close all

letterDataset = LetterDatasetClass()
trainSize = size(letterDataset.trainData);
testSize = size(letterDataset.testData);
assert(trainSize(1) == 16000, "Expected the dataset training to contain 16000 examples");
assert(trainSize(2) == 17, "Expected the dataset training to contain 16 features plus 1 target value");
assert(testSize(1) == 4000, "Expected the dataset training to contain 16000 examples");
assert(testSize(2) == 17, "Expected the dataset training to contain 16 features plus 1 target value");
disp("The dataset appears to be OK.");

%% Display sample's target values distribution to confirm it is equally distributed:
disp("Dataset distribution of examples:")
tabulate(table2array(letterDataset.allData(:,1)))
%% display correlation of attributes as a heatmap:
letterDataset.displayCorrelation()
%% Display a grid comparing the attributes by plotting attributes against each other.
letterDataset.displayScatterMatrix();