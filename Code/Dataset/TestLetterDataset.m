%
%% Test the dataset class:
% loads the dataset and checks the dimensions of the data are
% as expected.
%
% Displays all the plots coded for this dataset class.
%
clear
clc
clf
close all

letterDataset = LetterDatasetClass();
trainSize = size(letterDataset.trainTable);
testSize = size(letterDataset.testTable);
assert(trainSize(1) == 16000, "Expected the dataset training to contain 16000 examples");
assert(trainSize(2) == 17, "Expected the dataset training to contain 16 features plus 1 target value");
assert(testSize(1) == 4000, "Expected the dataset training to contain 16000 examples");
assert(testSize(2) == 17, "Expected the dataset training to contain 16 features plus 1 target value");
disp("The dataset appears to be OK.");

%%
% Display dataset information
disp(letterDataset);
disp("Training Table Summary:");
disp("=======================");
summary(letterDataset.trainTable);

%% Display sample's target values distribution to confirm it is equally distributed:
letterDataset.plotLetterDistribution(letterDataset.trainTable, "Distribution of Classes");
%% display correlation of attributes as a heatmap:
letterDataset.displayCorrelation(letterDataset.trainTable, "Correlation")
%% Display a grid comparing the attributes by plotting attributes against each other.
letterDataset.displayScatterMatrix(letterDataset.trainTable, "Scatter Matrix of Attributes");
%% Display Dataset PCA
letterDataset.plotPCA(letterDataset.trainTable, "Principle Component Analysis");
%% Display parallel coordinates plot of each class, value and feature
letterDataset.plotParallelCoordinates(letterDataset.trainTable, 'Parallel Coordinates Plot');