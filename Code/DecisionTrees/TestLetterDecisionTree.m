%% Test DecisionTreeClass
% Initialise
clear
clc
clf
close all
%% Test LetterDecisionTreeClass code works
% load letterDataset mat file, create tree class
load letterDatasetClass.mat
treeQuickTestClass = LetterDecisionTreeClass(letterDataset);
% display info during run:
treeQuickTestClass.debug = true;
% Test LetterDecisionTreeClass code works using  small set of hyperparameters
quickTestyperparameters = DTreeHyperparametersClass.getQuickTestRunInstance();
quickRunResults = treeQuickTestClass.performDTreeHyperameterAnalysis(quickTestyperparameters, "quickRunResults.csv");
disp(quickRunResults);

%% Run analysis on all columns, all parameters:
% load letterDataset mat file, create tree class
load letterDatasetClass.mat
treeAllFeaturesClass = LetterDecisionTreeClass(letterDataset);
treeAllFeaturesClass.debug = true;
% get an initial set of hyperparameters to try, the results allow a finer
% selection to be made later
allFeaturesHyperparameters = DTreeHyperparametersClass.getInstance();
treeAllFeatureResults = treeAllFeaturesClass.performDTreeHyperameterAnalysis(allFeaturesHyperparameters, "treeAllFeatureResults.csv");
disp(treeAllFeatureResults);
%% Plot results: deviance split criterion looks more accurate:
treeAllFeatureResults.plotCriteriaAccuracy("Accuracy by Split Criteria, all Parameters")

%
%% Feature selection try removing features
% Run the same hyperparameters as in the previous test, this time removing features identified as
% good candidates for removal during the dataset analysis
load letterDatasetClass.mat;
letterDataset.removeColumn("yBox");
letterDataset.removeColumn("xEgvy");
selectedFeaturesHyperparameters = DTreeHyperparametersClass.getInstance();
treeSelectFeatureClass = LetterDecisionTreeClass(letterDataset);
% display info during run:
treeSelectFeatureClass.debug = true;
treeSelectFeatureResults = treeSelectFeatureClass.performDTreeHyperameterAnalysis(selectedFeaturesHyperparameters, "treeSelectFeatureResults.csv");
disp(treeSelectFeatureResults);
%% Plot results: deviance split criterion looks more accurate, the time
% taken is not greatly reduced,the accuracies are slightly reduced when
% comparing the plot to the previous plot.
treeSelectFeatureResults.plotCriteriaAccuracy("Accuracy by Split Criteria (feature Selected)")

%
%% Perform analysiss on deviance split criteria
%
load letterDatasetClass.mat;
devianceHyperparameters = DTreeHyperparametersClass.getDevianceSplitCriteriaInstance();
treeAllFeatureClass = LetterDecisionTreeClass(letterDataset);
% display info during run:
treeAllFeatureClass.debug = true;
treeDevianceSplitResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(devianceHyperparameters, "treeSelectDevianceSplitResults.csv");
disp(treeDevianceSplitResults);
treeDevianceSplitResults.plotCriteriaAccuracy("Deviance Accuracy Split Criterion")
%% Plot Accuracy comparison
treeDevianceSplitResults.plotAccuracyTestTrainComparison()

%
%% Perform final test using optimal hyperparameters on complete dataset
%
load letterDatasetClass.mat;
finalHyperparameters = DTreeHyperparametersClass.getFinalHyperparameterInstance();
treeFinalFeatureClass = LetterDecisionTreeClass(letterDataset);
% display info during run:
treeFinalFeatureClass.debug = true;
treeFinalFeatureResults = treeFinalFeatureClass.performFinalDTreeHyperparameterAnalysis(finalHyperparameters, "treeFinalFeatureResults.csv");
disp(treeFinalFeatureResults);
%% Plot Accuracy comparison
treeFinalFeatureResults.plotAccuracyTestTrainComparison()

%{

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

%}