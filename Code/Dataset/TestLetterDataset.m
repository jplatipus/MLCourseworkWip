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
letterDatasetNormalised = LetterDatasetClass(true);
letterDatasets = [letterDatasetNotNormalised letterDatasetNormalised];

%% Display dataset analysis for both datasets: plots should be the same for 
% for both, console outputs min median and max are different for the
% dataset column values.
for letterDataset = letterDatasets
  normText = "(~normalised)";
  if letterDataset.isNormalised
    normText = "(normalised)";
  end
  fprintf("Dataset information %s\n", normText);
  trainSize = size(letterDataset.trainTable);
  testSize = size(letterDataset.testTable);
  assert(trainSize(1) == 16000, "Expected the dataset training to contain 16000 examples");
  assert(trainSize(2) == 17, "Expected the dataset training to contain 16 features plus 1 target value");
  assert(testSize(1) == 4000, "Expected the dataset training to contain 16000 examples");
  assert(testSize(2) == 17, "Expected the dataset training to contain 16 features plus 1 target value");
  disp("The dataset was successfully created.");
  letterDataset.displayDatasetInformation();