classdef LetterDecisionTreeClass < handle
%%
% Decision tree class
  properties
    debug = true;
    dataset;
    trainedClassifier;
    validationAccuracy;
  end
  
  methods
    %% Constructor
    % input : dataset, dataset class to use
    function obj = LetterDecisionTreeClass(dataset)
      obj.dataset = dataset;
    end
    
    function displayTree(obj, treeModel)
      %model = fitctree(obj.dataset.xTrain, obj.dataset.yTrain);
      view(treeModel, 'Mode', 'graph');
    end
    
    %
    % Fit the decision tree and return the model 
    %
    function treeModel = buildSimpleTree(obj)
      % Train a classifier
      % This code specifies all the classifier options and trains the classifier.
      [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      treeModel = fitctree(...
        x, ...
        y, ...
        'SplitCriterion', 'gdi', ...
        'MaxNumSplits', 100, ...
        'Surrogate', 'off', ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)));  
    end
    
    function trainSimpleTree(obj)
      [obj.trainedClassifier, obj.validationAccuracy] = vanillaTrainClassifier(obj.dataset.trainData);
    end
         
    % 
    % Good for large datasets: we have a largish one at 16000 examples for 
    % the train/validate, and 16 features.
    %
    function [letterDecisionTreeResults] = performDTreeHyperameterAnalysis(obj, hyperparameters)
      fprintf("Starting decision tree analysis...\n");
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = LetterDecisionTreeResults();
      resultsTable.startGatheringResults();
      for trainValidateProportion = hyperparameters.trainValidateProportions
        for maxNumSplit = hyperparameters.maxNumSplits
          for splitCriterion = hyperparameters.splitCriteria
            for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns
              % repeat holdout partition creation, build tree, predict this
              % number of times to get average:
              avgTrainAccuracy = 0.0;
              avgTestAccuracy = 0.0;
              startTime = cputime; 
              trainAccuracies = zeros(1, numberOfHoldOutRun);
              testAccuracies = zeros(1, numberOfHoldOutRun);
              accuracyIndex = 1;
              for holdOutTestRunCount = 1:numberOfHoldOutRun
                % calculate train and test subset indeces
                partition = cvpartition(numExamples, 'Holdout', 1.0 - trainValidateProportion);
                trainSubsetIdx = training(partition);
                testSubsetIdx = test(partition);
                [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
                % extract train and test subsets
                xTrain = x(trainSubsetIdx, :);
                yTrain = y(trainSubsetIdx, :);
                xTest = x(testSubsetIdx, :);
                yTest = y(testSubsetIdx, :);
                [trainAccuracy, testAccuracy] = obj.buildAndTestTree(xTrain, ...
                                      yTrain, xTest, yTest, ...
                                      splitCriterion, maxNumSplit, classNames);
                trainAccuracies(accuracyIndex) = trainAccuracy;
                testAccuracies(accuracyIndex) = testAccuracy;
                accuracyIndex = accuracyIndex + 1;
              end
              endTime = cputime;
              avgTrainAccuracy = mean(trainAccuracies);
              avgTestAccuracy = mean(testAccuracies);
              resultsTable.appendResult(trainValidateProportion, maxNumSplit, ...
                                       splitCriterion, numberOfHoldOutRun, ...
                                       avgTrainAccuracy, avgTestAccuracy, endTime - startTime);
            end
          end
        end
      end
      resultsTable.endGatheringResults();
      letterDecisionTreeResults = resultsTable;
      fprintf("Completed decision tree analysis\n");
    end % function

    %
    % Build a decision tree, and test it using the passed parameters 
    % The following parameters are the values passed directly to fitctree:
    %   xTrain, yTrain, splitCriterion, maxNumSplit, classNames
    % The following parameters are the values passed directly to predict:
    %   xTest is passed to predict
    %   yTest are the expected results.
    %
    function [trainingLoss, testLoss] = buildAndTestTree(obj, xTrain, yTrain, xTest, yTest, splitCriterion, maxNumSplit, classNames)
      % Grow the decision tree using the training subset
      treeModel = fitctree( ...
          xTrain, ...
          yTrain, ...
          'SplitCriterion', splitCriterion, ...
          'MaxNumSplits', maxNumSplit, ...
          'Surrogate', 'off', ...
          'ClassNames', classNames); 
      % predict
      [predictionResult, nodeNumbers] = predict(treeModel, xTest);
      % errors
      
      %numMisclass = sum(~strcmp(predictionResult,yTest))
      
      trainingLoss = loss(treeModel, xTrain, yTrain);
      testLoss = loss(treeModel, xTest, yTest);
      if obj.debug
        % display several prediction results
        %predictionResult(randsample(numel(predictionResult), 5))
        fprintf("Training loss: %0.02f. Test Loss: %0.02f\n", trainingLoss, testLoss);
        numMisclass = sum(~strcmp(predictionResult,yTest));
        fprintf("Misclassifications: %d\n", numMisclass);
      end
    end % function
    
  end % methods
  
  
end % class