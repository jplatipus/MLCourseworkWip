classdef NBayesClass < handle
  %NBAYESCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    randomSeed = 300;
    debug = true;
    % letter dataset used in this instance:
    dataset;
    % Matlab Model:
    nBayesModel;
    x;
    y; 
    xt; 
    yt;
    % distribution names used in the default (normal) distribution
    distNamesDefault = {'normal','normal','normal','normal','normal','normal','normal','normal','normal',...
      'normal','normal','normal','normal','normal','normal','normal'};
    % distribution names used in the kernel distribution
    distNamesKernel = {'kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel',...
      'kernel','kernel','kernel','kernel','kernel','kernel','kernel'};
  end
  
  methods
    
    %% Constructor
    % Several getInstance static convenience methods use this constructor.
    %
    function obj = NBayesClass(letterDataset)
      obj.dataset = letterDataset;
      rng(obj.randomSeed);
      [obj.x, obj.y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      [obj.xt, obj.yt] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
    end % constructor
    
    % Fit the model
    function fitModel(obj, distributionNames)
        model = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'DistributionNames', distributionNames);
      obj.nBayesModel = model;
    end % method
    
    % calculate the model's training and test loss
    function [trainingLoss, testLoss] = getModelLoss(obj)
        trainingLoss = loss(obj.nBayesModel, obj.x, obj.y);
        testLoss = loss(obj.nBayesModel, obj.xt, obj.yt);
    end
    
    % Set the Prior distribution to reflect the dataset's
    % training distribution of classes
    function nBayesModel = setPriorDistributionEmpirical(obj)
      Y = table2array(obj.y);
      freqDist = cell2table(tabulate(Y));
      prior = freqDist{:,3};
      obj.nBayesModel.Prior = prior;
      nBayesModel = obj.nBayesModel;
    end % function
    
    % Perform matlab's hyperparameter optimization search
    function nBayesModel = fitMatlabHyperparameterOptimization(obj)
      obj.nBayesModel = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus'));
      nBayesModel = obj.nBayesModel;
    end % function

    function letterNBayesResults = performHyperparameterSearch(obj, hyperparameters, resultsCsvFilename)
     fprintf("Starting decision tree analysis...\n");
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = LetterNBayesResults(resultsCsvFilename);
      resultsTable.startGatheringResults();      
      for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns
        % repeat holdout partition creation, build tree, predict this
        % number of times to get average:
        avgTrainAccuracy = 0.0;
        avgTestAccuracy = 0.0;
        startTime = cputime; 
        trainAccuracies = zeros(1, numberOfHoldOutRun);
        testAccuracies = zeros(1, numberOfHoldOutRun);
        misclassificationCounts = zeros(1, numberOfHoldOutRun);
        accuracyIndex = 1;
        for holdOutTestRunCount = 1:numberOfHoldOutRun
          for distributionName = hyperparameters.distributionNames
            for width = hyperparameters.kernelWidths
              % calculate train and test subset indeces
              partition = cvpartition(numExamples, 'Holdout', 0.2);
              trainSubsetIdx = training(partition);
              testSubsetIdx = test(partition);
              [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
              % extract train and test subsets
              xTrain = x(trainSubsetIdx, :);
              yTrain = y(trainSubsetIdx, :);
              xTest = x(testSubsetIdx, :);
              yTest = y(testSubsetIdx, :);
              [trainLoss, testLoss, misclassifiedCount] = obj.buildAndTestNBayes(xTrain, ...
                                    yTrain, xTest, yTest, ...
                                    distributionName, width, classNames);
              trainAccuracies(accuracyIndex) = 1 - trainLoss;
              testAccuracies(accuracyIndex) = 1 - testLoss;
              misclassificationCounts(accuracyIndex) = misclassifiedCount;
              accuracyIndex = accuracyIndex + 1;
            end % distributionName
          end % width
        end % holdOutTestRunCount
        endTime = cputime;
        avgTrainAccuracy = mean(trainAccuracies);
        avgTestAccuracy = mean(testAccuracies);
        avgMisclassificationCount = mean(misclassificationCounts);
        resultsTable.appendResult(trainValidateProportion, maxNumSplit, ...
                                 splitCriterion, numberOfHoldOutRun, ...
                                 avgTrainAccuracy, avgTestAccuracy, avgMisclassificationCount, size(yTest, 1), endTime - startTime);
      end % numberOfHoldOutRun        
      resultsTable.endGatheringResults();
      letterNBayesResults = resultsTable;
      fprintf("Completed NBayes analysis\n");      
    end % function
    
    function [trainLoss, testLoss, misclassifiedCount] = obj.buildAndTestNBayes(xTrain, ...
                                    yTrain, xTest, yTest, ...
                                    distributionName, width, classNames)
        TODO: set width for kernel, not normal distribution
        fit tree, predict, get loss
    end % function
    
    
  end %methods
  
  %
  % Static methods
  %
  methods(Static)

    % Get the default class instance for the given dataset
    function nBayesClassInstance = getDefaultInstance(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
      nBayesClassInstance.fitModel(nBayesClassInstance.distNamesDefault);
    end %function    
    
    % Get the kernel distribution class instance for the given dataset
    function nBayesClassInstance = getKernelInstance(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
      nBayesClassInstance.fitModel(nBayesClassInstance.distNamesKernel);
    end %function  
    
    % Ensures the nBayesClassInstance's distNamesDefault and 
    % distNamesKernel have the same number of entries as the letterDataset
    % has attributes
    function alignDistributionNames(nBayesClassInstance)
      letterDataset = nBayesClassInstance.dataset;
      attributeCount = size(letterDataset.trainTable);
      attributeCount = attributeCount(1,2);
      nBayesClassInstance.distNamesDefault = nBayesClassInstance.distNamesDefault(:,1:attributeCount - 1);
      nBayesClassInstance.distNamesKernel = nBayesClassInstance.distNamesKernel(:,1:attributeCount - 1);
    end
  end % static methods
end

