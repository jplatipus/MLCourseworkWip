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
letterDatasetNotNormalised = LetterDatasetClass(false);
letterDatasetStandardised = LetterDatasetClass(true);
letterDatasetStandardisedReducedFeatures = LetterDatasetClass(true);
letterDatasetStandardisedReducedFeatures.removeColumn("xBox");
letterDatasetStandardisedReducedFeatures.removeColumn("yBox");
letterDatasetStandardisedReducedFeatures.removeColumn("width");
letterDatasetStandardisedReducedFeatures.removeColumn("height");
letterDatasets = [letterDatasetNotNormalised letterDatasetStandardised letterDatasetStandardisedReducedFeatures];

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

function checkDataset(letterDataset, expectedTrainExamples, expectedTestExamples, expectedAttributes)
  normText = "(~normalised";
  if letterDataset.isStandardised
    normText = "(standardised";
  end
  if letterDataset.isRemovedFeature
    normText = normText + ", feature selection";
  end
  normText = normText + ")";
  fprintf("Dataset information %s\n", normText);
  trainSize = size(letterDataset.trainTable);
  testSize = size(letterDataset.testTable);
  assert(trainSize(1) == expectedTrainExamples, "Expected the dataset training to contain " + expectedTrainExamples + " examples");
  assert(trainSize(2) == expectedAttributes, "Expected the dataset training to contain " + expectedAttributes + " attributes");
  assert(testSize(1) == expectedTestExamples, "Expected the dataset training to contain " + expectedTrainExamples + " examples");
  assert(testSize(2) == expectedAttributes, "Expected the dataset training to contain " + expectedAttributes + " attributes");
  letterDataset.displayDatasetInformation();
  disp("The dataset " + normText + " was successfully created.");
end