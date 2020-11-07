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
treeAllFeatureClass.debug = true;
%% Test LetterDecisionTreeClass code works
hyperparameters = DTreeHyperparametersClass.getQuickTestRunInstance();
quickRunResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(hyperparameters, "quickRunResults.csv");
disp(quickRunResults);

%% Run analysis on all columns, all parameters (displays debug info):
hyperparameters = DTreeHyperparametersClass.getInstance();
treeAllFeatureResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(hyperparameters, "treeAllFeatureResults.csv");
disp(treeAllFeatureResults);
% Plot results: deviance split criterion looks more accurate:
treeAllFeatureResults.plotCriteriaAccuracy("Accuracy by Split Criteria, all Parameters")

%
%% Feature selection try removing features
%
load letterDatasetClass.mat;
letterDataset.removeColumn("yBox");
letterDataset.removeColumn("xEgvy");
hyperparameters = DTreeHyperparametersClass.getInstance();
treeSelectFeatureClass = LetterDecisionTreeClass(letterDataset);
treeSelectFeatureResults = treeSelectFeatureClass.performDTreeHyperameterAnalysis(hyperparameters, "treeSelectFeatureResults.csv");
disp(treeSelectFeatureResults);
% Plot results: deviance split criterion looks more accurate:
treeSelectFeatureResults.plotCriteriaAccuracy("Accuracy by Split Criteria (feature Selected)")

%
%% Perform analysiss on deviance split criteria
%
load letterDatasetClass.mat;
devianceHyperparameters = DTreeHyperparametersClass.getDevianceSplitCriteriaInstance();
treeAllFeatureClass = LetterDecisionTreeClass(letterDataset);
treeDevianceSplitResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(devianceHyperparameters, "treeSelectDevianceSplitResults.csv");
disp(treeDevianceSplitResults);
treeDevianceSplitResults.plotCriteriaAccuracy("Deviance Accuracy Split Criterion")

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