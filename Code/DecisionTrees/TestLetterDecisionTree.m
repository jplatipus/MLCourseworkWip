%% Test DecisionTreeClass
% Initialise
clear
clc
clf
close all

%% load dataset mat file & configure tree
load letterDatasetClass.mat
decisionTree = LetterDecisionTreeClass(letterDataset);
treeModel = decisionTree.buildTree()

%% display tree
decisionTree.displayTree(treeModel)

%% Cross validation loss: 37.44%, or approx 62% accurrate, the paper mentions 80% accuracy.
cvTree = crossval(treeModel, 'KFold', 5)
% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(cvTree);

% Compute validation accuracy
[x, y] = letterDataset.extractXYFromTable(letterDataset.trainTable);
validationAccuracy = 1 - kfoldLoss(cvTree, 'LossFun', 'ClassifError');
yTrain = categorical(table2cell(y));
% Plot confusion matrix
matrix = confusionmat(yTrain, validationPredictions);
confusionchart(matrix)