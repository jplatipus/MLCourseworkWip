classdef LetterDecisionTreeClass < handle
%%
% Decision tree class: the code to perform the hyperparameter search, and
% to create the final model is in this class.
% It takes a dataset as part of the creation so that the search results cn
% be compared between the original feature values, normalised, and
% normalised with some features removed.
  properties
    debug = false;
    dataset;
    defaultRandomStream;
    dTreeRandomStream;
    trainedClassifier;
    validationLoss;
  end
  
  methods
    %% Constructor
    % input : dataset, dataset class to use
    % create 2 random stream: defaultRandomStream is used for splitting the train and
    % validate data, dTreeRandomStream is used to train the decision trees.
    function obj = LetterDecisionTreeClass(dataset)
      obj.dataset = dataset;
      obj.defaultRandomStream = RandStream('mt19937ar');
      obj.dTreeRandomStream = RandStream('mt19937ar');
    end
    
    %%
    % Displays the tree, wrapped here so that additional settings can be
    % added.
    function displayTree(obj, treeModel)
      view(treeModel, 'Mode', 'graph');
    end
            
    %% 
    % performs a hyperparameter search on the given dataset.
    % The results are saved using the given csv (tab delimited) filename.
    % Returns the results
    %
    function [letterDecisionTreeResults] = performDTreeHyperameterAnalysis(obj, hyperparameters, resultsCsvFilename)
      fprintf("Starting decision tree analysis...\n");
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = LetterDecisionTreeResults(resultsCsvFilename);
      resultsTable.startGatheringResults();
      rowNumber = 1;
      % iterate over the hyperparameters
      for splitCriterion = hyperparameters.splitCriteria
        for maxNumSplit = hyperparameters.maxNumSplits
          for minLeafSize = hyperparameters.minLeafSizes
            for minParentSize = hyperparameters.minParentSizes
              for numberOfFold = hyperparameters.numberOfFolds
                % repeat fold partition creation, build tree, predict this
                % number of times to get average:
                startTime = cputime; 
                predictTimes = zeros(1, numberOfFold);
                trainLosses = zeros(1, numberOfFold);
                testLosses = zeros(1, numberOfFold);
                accuracies = zeros(1, numberOfFold);
                precisions = zeros(1, numberOfFold);
                recalls = zeros(1, numberOfFold);
                f1s = zeros(1, numberOfFold);
                LossIndex = 1;
                for foldTestRunCount = 1:numberOfFold
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
                  % train and test predict the partitions
                  [trainLoss, testLoss, misclassifiedCount, ...
                      accuracy, ...
                      precision, recall, f1, predictTime, model] ...
                    = obj.buildAndTestTree(xTrain, ...
                                        yTrain, xTest, yTest, ...
                                        minLeafSize, minParentSize, ...
                                        splitCriterion, maxNumSplit, classNames, rowNumber);
                  % gather this fold's results
                  trainLosses(LossIndex) =trainLoss;
                  testLosses(LossIndex) =testLoss;
                  accuracies(LossIndex) = accuracy;
                  precisions(LossIndex) = precision;
                  recalls(LossIndex) = recall;
                  f1s(LossIndex) = f1;
                  predictTimes(LossIndex) = predictTime;
                  LossIndex = LossIndex + 1;
                end % foldTestRunCount
                endTime = cputime;
                avgTrainLoss = mean(trainLosses);
                avgTestLoss = mean(testLosses);
                avgPrecision = mean(precisions);
                avgAccuracy = mean(accuracies);
                avgRecall = mean(recalls);
                avgF1 = mean(f1s);
                % save mean results in the csv file
                resultsTable.appendResult(minLeafSize, minParentSize, maxNumSplit, ...
                                         splitCriterion, numberOfFold, ...
                                         avgTrainLoss, avgTestLoss, ...
                                         avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                                         size(yTest, 1), endTime - startTime,... 
                                         mean(predictTimes), rowNumber);
                % increment row number: used as a
                % random seed for growing a decision tree
                rowNumber = rowNumber + 1;
              end % numberOfFold
            end % minParentSize
          end % minLeafSize
        end % splitCriterion
      end % maxNumSplit
      resultsTable.endGatheringResults();
      letterDecisionTreeResults = resultsTable;
      fprintf("Completed decision tree analysis\n");
    end % function

    %
    % Perform final analysis, using the previously unseen dataset
    function [letterDecisionTreeResults, dTreeModel, predictionResult] = performFinalDTreeHyperparameterAnalysis(obj, hyperparameters, resultsFinalCsvFilename)
            fprintf("Starting final decision tree analysis...\n");
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = LetterDecisionTreeResults(resultsFinalCsvFilename);
      resultsTable.startGatheringResults();
      % get hyperparameter values to use
      maxNumSplit = hyperparameters.maxNumSplits(1);
      splitCriterion = hyperparameters.splitCriteria(1);
      minLeafSize = hyperparameters.minLeafSizes(1);
      minParentSize = hyperparameters.minParentSizes(1);
      % create train and test sets from complete dataset
      % extract train and test sets using the TRAIN and TEST TABLE
      [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      xTrain = x;
      yTrain = y;
      [x, y] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
      xTest = x;
      yTest = y;
      startTime = cputime; 
      % train model and predict using the training set, predict on test set:
      [trainLoss, testLoss, misclassifiedCount, ...
        accuracy, ...
        precision, recall, f1, predictTime, dTreeModel, predictionResult] = obj.buildAndTestTree(xTrain, ...
                            yTrain, xTest, yTest, ...
                            minLeafSize, minParentSize, ...
                            splitCriterion, maxNumSplit, classNames,...
                            hyperparameters.randomSeed);
      endTime = cputime;
      resultsTable.appendResult(minLeafSize, minParentSize, maxNumSplit, ...
                               splitCriterion, 1, ...
                               trainLoss, testLoss, ...
                               accuracy, ...
                                precision, recall, f1, ...
                             size(yTest, 1), endTime - startTime, predictTime,...
                             hyperparameters.randomSeed);

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
              precision, recall, f1, predictTime, treeModel, predictionResult] = ...
            buildAndTestTree(obj, ...
                              xTrain, yTrain, xTest, yTest, ...
                              minLeafSize, minParentSize, ...
                              splitCriterion, maxNumSplit, classNames,...
                              dTreeRandomSeed)
      % Grow the decision tree using the training subset
      RandStream.setGlobalStream(obj.dTreeRandomStream);
      rng(dTreeRandomSeed);
      treeModel = fitctree( ...
          xTrain, ...
          yTrain, ...
          'SplitCriterion', splitCriterion, ...
          'MaxNumSplits', maxNumSplit, ...
          'Surrogate', 'off', ...
          'MinLeafSize', minLeafSize,...
          'MinParentSize', minParentSize,...
          'ClassNames', classNames); 
      RandStream.setGlobalStream(obj.defaultRandomStream);
      % predict
      startTime = cputime;
      [predictionResult, nodeNumbers] = predict(treeModel, xTest);
      predictTime = cputime - startTime;
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