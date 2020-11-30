classdef LetterDecisionTreeClass < handle
%%
% Decision tree class
  properties
    debug = false;
    dataset;
    trainedClassifier;
    validationLoss;
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
      [obj.trainedClassifier, obj.validationLoss] = vanillaTrainClassifier(obj.dataset.trainData);
    end
         
    % 
    % Good for large datasets: we have a largish one at 16000 examples for 
    % the train/validate, and 16 features.
    %
    function [letterDecisionTreeResults] = performDTreeHyperameterAnalysis(obj, hyperparameters, resultsCsvFilename)
      fprintf("Starting decision tree analysis...\n");
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = LetterDecisionTreeResults(resultsCsvFilename);
      resultsTable.startGatheringResults();
      for trainValidateProportion = hyperparameters.trainValidateProportions
        for maxNumSplit = hyperparameters.maxNumSplits
          for splitCriterion = hyperparameters.splitCriteria
            for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns
              % repeat holdout partition creation, build tree, predict this
              % number of times to get average:
              startTime = cputime; 
              trainLosses = zeros(1, numberOfHoldOutRun);
              testLosses = zeros(1, numberOfHoldOutRun);
              accuracies = zeros(1, numberOfHoldOutRun);
              precisions = zeros(1, numberOfHoldOutRun);
              recalls = zeros(1, numberOfHoldOutRun);
              f1s = zeros(1, numberOfHoldOutRun);
              LossIndex = 1;
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
                [trainLoss, testLoss, misclassifiedCount, ...
                    accuracy, ...
                    precision, recall, f1] ...
                = obj.buildAndTestTree(xTrain, ...
                                      yTrain, xTest, yTest, ...
                                      splitCriterion, maxNumSplit, classNames);
                trainLosses(LossIndex) =trainLoss;
                testLosses(LossIndex) =testLoss;
                accuracies(LossIndex) = accuracy;
                precisions(LossIndex) = precision;
                recalls(LossIndex) = recall;
                f1s(LossIndex) = f1;
                LossIndex = LossIndex + 1;
              end % holdOutTestRunCount
              endTime = cputime;
              avgTrainLoss = mean(trainLosses);
              avgTestLoss = mean(testLosses);
              avgPrecision = mean(precisions);
              avgAccuracy = mean(accuracies);
              avgRecall = mean(recalls);
              avgF1 = mean(f1s);
              resultsTable.appendResult(trainValidateProportion, maxNumSplit, ...
                                       splitCriterion, numberOfHoldOutRun, ...
                                       avgTrainLoss, avgTestLoss, ...
                                       avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                                       size(yTest, 1), endTime - startTime);
            end % numberOfHoldOutRun
          end % splitCriterion
        end % maxNumSplit
      end % trainValidateProportion
      resultsTable.endGatheringResults();
      letterDecisionTreeResults = resultsTable;
      fprintf("Completed decision tree analysis\n");
    end % function

    %%
    % Perform final analysis, usimng the previously unseen dataset
    function [letterDecisionTreeResults] = performFinalDTreeHyperparameterAnalysis(obj, hyperparameters, resultsFinalCsvFilename)
            fprintf("Starting final decision tree analysis...\n");
      rng(hyperparameters.randomSeed);
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = LetterDecisionTreeResults(resultsFinalCsvFilename);
      resultsTable.startGatheringResults();
      for maxNumSplit = hyperparameters.maxNumSplits
        for splitCriterion = hyperparameters.splitCriteria
          for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns
            startTime = cputime; 
            % create train and test sets from complete dataset
            % extract train and test sets
            [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
            xTrain = x;
            yTrain = y;
            [x, y] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
            xTest = x;
            yTest = y;
            trainValidateProportion = size(yTrain) / (size(yTrain ) + size(yTest));
            [trainLoss, testLoss, misclassifiedCount, ...
              accuracy, ...
              precision, recall, f1] = obj.buildAndTestTree(xTrain, ...
                                  yTrain, xTest, yTest, ...
                                  splitCriterion, maxNumSplit, classNames);
            endTime = cputime;
            resultsTable.appendResult(trainValidateProportion, maxNumSplit, ...
                                     splitCriterion, 1, ...
                                     trainLoss, testLoss, ...
                                     accuracy, ...
                                      precision, recall, f1, ...
                                   size(yTest, 1), endTime - startTime);
          end
        end
      end
      resultsTable.endGatheringResults();
      letterDecisionTreeResults = resultsTable;
      fprintf("Completed final decision tree analysis\n");
    end %function
    
    %
    % Build a decision tree, and test it using the passed parameters 
    % The following parameters are the values passed directly to fitctree:
    %   xTrain, yTrain, splitCriterion, maxNumSplit, classNames
    % The following parameters are the values passed directly to predict:
    %   xTest is passed to predict
    %   yTest are the expected results.
    %
    function [trainingLoss, testLoss, misclassifiedCount, accuracy, ...
              precision, recall, f1] = ...
            buildAndTestTree(obj, ...
                              xTrain, yTrain, xTest, yTest, ...
                              splitCriterion, maxNumSplit, classNames)
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
      yTestArray = table2array(yTest);
      misclassifiedCount = sum(predictionResult ~= categorical(yTestArray));
      testLoss = loss(treeModel, xTest, yTest);
      [accuracy, precision, recall, ...
        f1] = CalcUtil.calculateMeasuresFromExpectPredict(yTestArray, predictionResult);
      trainingLoss = loss(treeModel, xTrain, yTrain);
      
      if obj.debug
        % display random 5 predictions
        predictionResult(randsample(numel(predictionResult), 5))
        percentMisclassified = (misclassifiedCount / size(yTest, 1)) * 100;
        fprintf("Misclassified %d entries out of %d. %0.04f pct\n", misclassifiedCount, size(yTest, 1), percentMisclassified);
        fprintf("Training loss: %0.02f. Test Loss: %0.02f\n", trainingLoss, testLoss);
      end
    end % function
    
  end % methods
  
  
end % class