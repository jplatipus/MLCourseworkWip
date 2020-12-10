%% Create the dataset class instances:
% loads the dataset csv file and checks the dimensions of the data are
% as expected.
%
% Creates 3 workspace variables that should be saved into
% letterDatasetClass.mat:
% - letterDatasetNotNormalised: the original data 
% - letterDatasetNormalised: the data normalised using z-score
%                             normalisation
% - letterDatasetNormalisedReducedFeatures: also normalised, highly
%                             correlated features removed.
clear
clc
clf
close all
letterDatasetNotNormalised = LetterDatasetClass(false);
letterDatasetNormalised = LetterDatasetClass(true);
letterDatasetNormalisedReducedFeatures = LetterDatasetClass(true);
letterDatasetNormalisedReducedFeatures.removeColumn("xBox");
letterDatasetNormalisedReducedFeatures.removeColumn("yBox");
letterDatasetNormalisedReducedFeatures.removeColumn("width");
letterDatasetNormalisedReducedFeatures.removeColumn("height");
letterDatasets = [letterDatasetNotNormalised letterDatasetNormalised letterDatasetNormalisedReducedFeatures];

%% Display dataset analysis for both datasets: plots should be the same for 
% for both, console outputs min median and max are different for the
% dataset column values.
for letterDataset = letterDatasets
  expectedTrainExamples = 16000;
  expectedTestExamples = 4000;
  expectedAttributes = 17;
  if letterDataset.isRemovedFeature
    expectedAttributes = 13;
  end
  checkDataset(letterDataset, expectedTrainExamples, expectedTestExamples, expectedAttributes)
end

% 
% Checks the dataset examples in the trin and test are the correct size.
% Displays dataset information: mean median and max of each feature.
%
function checkDataset(letterDataset, expectedTrainExamples, expectedTestExamples, expectedAttributes)
  normText = "(~normalised";
  if letterDataset.isNormalised
    normText = "(normalised";
  end
  if letterDataset.isRemovedFeature
    normText = normText + ", feature selection";
  end
  normText = normText + ")";
  fprintf("\n\nDataset information %s:\n" + ...
              "===================\n", normText);
  trainSize = size(letterDataset.trainTable);
  testSize = size(letterDataset.testTable);
  assert(trainSize(1) == expectedTrainExamples, "Expected the dataset training to contain " + expectedTrainExamples + " examples");
  assert(trainSize(2) == expectedAttributes, "Expected the dataset training to contain " + expectedAttributes + " attributes");
  assert(testSize(1) == expectedTestExamples, "Expected the dataset training to contain " + expectedTrainExamples + " examples");
  assert(testSize(2) == expectedAttributes, "Expected the dataset training to contain " + expectedAttributes + " attributes");
  letterDataset.displayDatasetInformation();
  fprintf("The dataset " + normText + " was successfully created.\n"+ ...
          "______________________________________________________________");
end