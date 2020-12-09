%% Test DecisionTreeClass
% Initialise
clear
clc
clf
close all
%% Test LetterDecisionTreeClass code works
% load letterDataset mat file, create tree class
load letterDatasetClass.mat
treeQuickTestClass = LetterDecisionTreeClass(letterDatasetStandardised);
% display info during run:
treeQuickTestClass.debug = true;
% Test LetterDecisionTreeClass code works using  small set of hyperparameters
quickTestyperparameters = DTreeHyperparametersClass.getQuickTestRunInstance();
quickRunResults = treeQuickTestClass.performDTreeHyperameterAnalysis(quickTestyperparameters, "quickRunResults.csv");
disp(quickRunResults);

%% Run analysis on all columns, all parameters:
% load letterDataset mat file, create tree class
load letterDatasetClass.mat
%% Perform analysis on unnormalised dataset (takes time)
treeAllFeatureResultsNotNormalised = allFeatureAnalysis(letterDatasetNotNormalised);
%% Perform analysis on normalised dataset (takes time)
treeAllFeatureResultsNormalised = allFeatureAnalysis(letterDatasetStandardised);
%% Display the results of test and train Loss on normalised and unnormalised 
% datasets: From this we can see that normalisation has no effect
displayFeatureResults(treeAllFeatureResultsNotNormalised, "Loss by Split Criteria, all Parameters (~normalised)" );
displayFeatureResults(treeAllFeatureResultsNormalised, "Loss by Split Criteria, all Parameters (normalised)");

%
%% Feature selection try removing features
% Run the same hyperparameters as in the previous test, this time removing features identified as
% good candidates for removal during the dataset analysis
load letterDatasetClass.mat;
%% Perform analysis on unnormalised dataset (takes time)
selectedFeatureResults = selectedFeatureAnalysis(letterDatasetStandardised);
%% Plot results: deviance split criterion looks more accurate, the time
% taken is not greatly reduced,the Losses are slightly increased when
% comparing the plot to the plot with all the features.
displayFeatureResults(selectedFeatureResults, "Loss by Split Criteria, selected Parameters (normalised)" );

%
%% Perform analysiss on deviance split criterion, as this criterion is consistently the most accurate.
%
load letterDatasetClass.mat;
devianceHyperparameters = DTreeHyperparametersClass.getDevianceSplitCriteriaInstance();
treeAllFeatureClass = LetterDecisionTreeClass(letterDatasetStandardised);
% display info during run:
treeAllFeatureClass.debug = true;
treeDevianceSplitResults = treeAllFeatureClass.performDTreeHyperameterAnalysis(devianceHyperparameters, "treeSelectDevianceSplitResults.csv");
disp(treeDevianceSplitResults);
treeDevianceSplitResults.plotCriteriaLoss("Deviance Loss Split Criterion");
%% Plot Loss comparison
treeDevianceSplitResults.plotLossTestTrainComparison("Deviance Split Criterion Loss by Result Row")

%
%% Perform final test using optimal hyperparameters on complete dataset
%
load letterDatasetClass.mat;
finalHyperparameters = DTreeHyperparametersClass.getFinalHyperparameterInstance();
treeFinalFeatureClass = LetterDecisionTreeClass(letterDatasetStandardised);
% display info during run:
treeFinalFeatureClass.debug = true;
treeFinalFeatureResults = treeFinalFeatureClass.performFinalDTreeHyperparameterAnalysis(finalHyperparameters, "treeFinalFeatureResults.csv");
disp(treeFinalFeatureResults);
%% Plot Loss comparison
treeFinalFeatureResults.plotLossTestTrainComparison("Final Loss by Result Row")

%% Functions defined for the initial tests (allFeatureAnalysis, selectedFeatureAnalysis) 
% so that they can be run on normalised and unnormalised datasets

%%
% Perform decision hyperparameter analysis on all features
function treeAllFeatureResults = allFeatureAnalysis(letterDataset)
  treeAllFeaturesClass = LetterDecisionTreeClass(letterDataset);
  treeAllFeaturesClass.debug = true;
  % get an initial set of hyperparameters to try, the results allow a finer
  % selection to be made later
  allFeaturesHyperparameters = DTreeHyperparametersClass.getInstance();
  treeAllFeatureResults = treeAllFeaturesClass.performDTreeHyperameterAnalysis(allFeaturesHyperparameters, "treeAllFeatureResults.csv");
end

%% Display test / train Loss plot (the plot is used to select the 
% most accurate split criterion)
function displayFeatureResults(featureResults, plotTitle)
  disp(featureResults);
  %% Plot results: deviance split criterion looks more accurate:
  featureResults.plotCriteriaLoss(plotTitle)
end

%%
% Perform decision hyperparameter analysis on selected features
function treeSelectFeatureResults = selectedFeatureAnalysis(letterDataset)
  letterDataset.removeColumn("yBox");
  letterDataset.removeColumn("xEgvy");
  selectedFeaturesHyperparameters = DTreeHyperparametersClass.getInstance();
  treeSelectFeatureClass = LetterDecisionTreeClass(letterDataset);
  % display info during run:
  treeSelectFeatureClass.debug = true;
  treeSelectFeatureResults = treeSelectFeatureClass.performDTreeHyperameterAnalysis(selectedFeaturesHyperparameters, "treeSelectFeatureResults.csv");
  disp(treeSelectFeatureResults);
end

%{

%% display tree
treeModel = decisionTree.buildSimpleTree()
decisionTree.displayTree(treeModel)
%% Cross validation loss: 37.44%, or approx 62% accurrate, the paper mentions 80% Loss.
cvTree = crossval(treeModel, 'KFold', 5)
% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(cvTree);

% Compute validation Loss
[x, y] = letterDataset.extractXYFromTable(letterDataset.trainTable);
validationLoss = 1 - kfoldLoss(cvTree, 'LossFun', 'ClassifError');
yTrain = categorical(table2cell(y));
% Plot confusion matrix
matrix = confusionmat(yTrain, validationPredictions);
confusionchart(matrix)

%}