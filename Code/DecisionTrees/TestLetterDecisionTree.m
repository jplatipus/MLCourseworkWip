%% Test DecisionTreeClass
% Initialise
clear
clc
clf
close all
%
%% load letterDataset mat file, create tree class
%
load letterDatasetClass.mat
treeAllFeatureClass = LetterDecisionTreeClass(letterDataset);

%% Test LetterDecisionTreeClass code works
hyperparameters = DTreeHyperparametersClass.getQuickTestRunInstance();
quickRunResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(hyperparameters, "quickRunResults.csv");
disp(quickRunResults);

%% Run analysis on all columns, all parameters:
hyperparameters = DTreeHyperparametersClass.getInstance();
treeAllFeatureResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(hyperparameters, "treeAllFeatureResults.csv");
disp(treeAllFeatureResults);
% Plot results: gdi split criterion looks more accurate:
treeAllFeatureResults.plotCriteriaAccuracy("Accuracy by Split Criteria, all Parameters")

%
%% Feature selection try removing features
%
load letterDatasetClass.mat;
letterDataset.removeColumn("yBox");
letterDataset.removeColumn("xEgvy");
letterDataset.removeColumn("xBox");
hyperparameters = DTreeHyperparametersClass.getInstance();
treeSelectFeatureClass = LetterDecisionTreeClass(letterDataset);
treeSelectFeatureResults = treeSelectFeatureClass.performDTreeHyperameterAnalysis(hyperparameters, "treeSelectFeatureResults.csv");
disp(treeSelectFeatureResults);
% Plot results: gdi split criterion looks more accurate:
treeSelectFeatureResults.plotCriteriaAccuracy("Accuracy by Split Criteria (feature Selected)")

%
%% Perform analysiss on gdi split criteria
%
load letterDatasetClass.mat;
gdiHyperparameters = DTreeHyperparametersClass.getGdiSplitCriteriaInstance();
treeAllFeatureClass = LetterDecisionTreeClass(letterDataset);
treeGdiSplitResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(gdiHyperparameters, "treeSelectGdiSplitResults.csv");
disp(treeGdiSplitResults);
treeGdiSplitResults.plotCriteriaAccuracy("GDI Accuracy Split Criterion")

%
%% Perform final test using optimal hyperparameters on complete dataset
%
load letterDatasetClass.mat;
finalHyperparameters = DTreeHyperparametersClass.getFinalHyperparameterInstance();
treeFinalFeatureClass = LetterDecisionTreeClass(letterDataset);
treeFinalFeatureClass.debug = true;
treeFinalFeatureResults = treeFinalFeatureClass.performDTreeHyperameterAnalysis(finalHyperparameters, "treeFinalFeatureResults.csv");
disp(treeFinalFeatureResults);

%% display tree
treeModel = decisionTree.buildSimpleTree()
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