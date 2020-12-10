classdef RandomForestClass < handle
  % This class is used to perform the hyperparameter search for the random
  % forest method. The functionality to create the final instance of the
  % model is also present.
  
  properties
    % parameter values to try in different scenarios:
    hyperparameters;
    % random stream used for partitioning the train / test data:
    defaultRandomStream;
    % random stream used for growing the forest:
    treeBaggerRandomStream;
    debug = false;
  end
  
  methods
    %
    % Constructor: initialise random streams for bagging and forest
    % training
    function obj = RandomForestClass(rfHyperparameters)
      obj.hyperparameters = rfHyperparameters;
      obj.defaultRandomStream = RandStream('mt19937ar');
      obj.treeBaggerRandomStream = RandStream('mt19937ar');
      RandStream.setGlobalStream(obj.defaultRandomStream)
    end
    
    %
    % Perform hyperparameter search,
    % save results to csv file, return results (file contents) wrapped in a class
    %
    function rfResults = performAnalysis(obj, letterDataset, csvFilename)
      rfResults = RFResults(csvFilename);
      rfResults.startGatheringResults();
      fprintf("Starting Random Forest analysis...\n");
      rng(obj.hyperparameters.randomSeed);
      numExamples = size(letterDataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(letterDataset.validClassValues));
      
      [x, y] = letterDataset.extractXYFromTable(letterDataset.trainTable);
      % calculate prior distribution of classes based on training dataset
      yAsArray = table2array(y);
      freqDist = cell2table(tabulate(yAsArray));
      priorDistribution = freqDist{:,3}/100;
      rng(obj.hyperparameters.randomSeed);
      % start initialising the treebagger random number generator seed.
      randomTreeBagSeed = 1;
      for numTree = obj.hyperparameters.trees
        for numFeature = obj.hyperparameters.features
          for numFolds = obj.hyperparameters.folds
            % initialise arrays to hold results (1 per fold)
            errTrain = zeros(size(numFolds,2),1);
            errValid = zeros(size(numFolds,2),1);
            errOob = zeros(size(numFolds,2),1);
            accuracies = zeros(size(numFolds,2),1);
            precisions = zeros(size(numFolds,2),1);
            recalls = zeros(size(numFolds,2),1);
            f1s = zeros(size(numFolds,2),1);
            predictTimes = zeros(size(numFolds,2),1);
            startTime = cputime;
            % for each fold train & predict using bagging using
            % hyperparameters selected for this loop.
            for foldCount = 1:numFolds
              % calculate train and test subset indeces
              partition = cvpartition(numExamples, 'Holdout', 0.2);
              trainSubsetIdx = training(partition);
              testSubsetIdx = test(partition);
              % extract train and test subsets
              xTrain = x(trainSubsetIdx, :);
              yTrain = y(trainSubsetIdx, :);
              xTest = x(testSubsetIdx, :);
              yTest = y(testSubsetIdx, :);
              [trainingErr, testErr, oobErr, misclassifiedCount, ...
                    model, accuracy, precision, recall, f1, predictTime] = ...
                   obj.buildAndTestTreeBagger(numTree, xTrain, yTrain, xTest, yTest, ...
                                priorDistribution, numFeature, randomTreeBagSeed);  
              % gather the results
              errTrain(foldCount) = trainingErr;
              errValid(foldCount) = testErr;
              errOob(foldCount)  = oobErr;
              accuracies(foldCount) = accuracy;
              precisions(foldCount) = precision;
              recalls(foldCount) = recall;
              f1s(foldCount) = f1;
              predictTimes(foldCount) = predictTime;
              if obj.debug
                % check bag is different, display train/predict result
                fprintf("\tnumTree %d numFeature %d trainErr %0.04f testErr %0.04f "+ ...
                        "oobErr %0.04f fold %d  of %d predictTime %0.04f" + ...
                        "\tSamples: %s %s %s %s %s\n", ...
                        numTree, numFeature, errTrain(foldCount), errValid(foldCount), ...
                        errOob(foldCount),foldCount, numFolds, predictTime, ...
                        string(table2array(yTrain(1,1))), ...
                        string(table2array(yTrain(2,1))), ...
                        string(table2array(yTrain(3,1))), ...
                        string(table2array(yTrain(4,1))), ...
                        string(table2array(yTrain(5,1))));
              end %debug
            end %foldCount
            elapsedTime = cputime - startTime;
            % save mean results for all folds to csv file
            rfResults.appendResult(numFolds, numTree, numFeature, randomTreeBagSeed, mean(errTrain), ...
              mean(errValid), mean(errOob), ...
              mean(accuracies), mean(precisions), mean(recalls), mean(f1s), ...
              size(yTest, 1), elapsedTime, mean(predictTimes));
            % next set of folds use a different random tree
            randomTreeBagSeed = randomTreeBagSeed + 1;
          end % folds
        end % feature
      end % tree
      rfResults.endGatheringResults();
    end % function
    
    %
    % grows the tree ensemble, setting the global random number generator 
    % treeBaggerRandomStream with the randomTreeBaggerSeed.
    % restores global random number generator to the defaultRandomStream
    % before exit.
    % 
    % creates the random forest, trains it and predicts it using the given
    % parameters.
    function [trainingLoss, testLoss, oobErr, misclassifiedCount, ...
                    model, accuracy, precision, recall, f1, predictTime] = ...
              buildAndTestTreeBagger(obj, numTree, xTrain, yTrain, xTest, yTest, ...
                                    priorDistribution, numFeature, randomTreeBaggerSeed) 
      % use the random number generator instance for growing the ensemble
      RandStream.setGlobalStream(obj.treeBaggerRandomStream);
      rng(randomTreeBaggerSeed);
      model = TreeBagger(numTree, xTrain, yTrain, 'OOBPrediction','On',...
                              'Method','classification', ...
                              'Prior', priorDistribution, ...
                              'NumPredictorsToSample', numFeature); 
      trainingLoss = mean(error(model, xTrain, yTrain));                              
      testLoss = mean(error(model, xTest, yTest));
      oobErr = mean(oobError(model));
      % restore the random number generator instance to the "default" one
      RandStream.setGlobalStream(obj.defaultRandomStream);
      % predict using given xTest
      startTime = cputime;
      [predictionResult, ~] = predict(model, xTest);
      predictTime = cputime - startTime;
      % errors
      misclassifiedCount = sum(predictionResult ~= categorical(table2array(yTest)));

      [accuracy, precision, recall, ...
        f1] = CalcUtil.calculateMeasuresFromExpectPredict(table2array(yTest), predictionResult);
      if obj.debug
        % display random 5 predictions, check loss and misclassified
        % percent match
        %predictionResult(randsample(numel(predictionResult), 5))
        percentMisclassified = (misclassifiedCount / size(yTest, 1)) * 100;
        fprintf("Misclassified %d entries out of %d. %0.04f pct", misclassifiedCount, size(yTest, 1), percentMisclassified);
        fprintf("\tTraining loss: %0.02f. Test Loss: %0.02f Oob Error: %0.02f", trainingLoss, testLoss, oobErr);
        fprintf("\tPrecision: %0.04f Recall: %0.04f Accuracy: %0.04f F1: %0.04f\n", ...
                 precision, recall, accuracy, f1);      
      end % debug        
    end % function  
      
  end %methods
    
    methods(Static)
      
    % build final tree bagger model
    function [treeBagger, trainTime] = buildFinalRandomForest(numberOfTrees, ...
        numberOfFeatures, randomSeed, xTrain, yTrain)
      % calculate prior distribution of classes based on training dataset
      yAsArray = table2array(yTrain);
      freqDist = cell2table(tabulate(yAsArray));
      priorDistribution = freqDist{:,3}/100;
      rng(randomSeed);
      startTime = cputime;
      treeBagger = TreeBagger(numberOfTrees, xTrain, yTrain, 'OOBPrediction','On',...
                              'Method','classification', ...
                              'Prior', priorDistribution, ...
                              'NumPredictorsToSample', numberOfFeatures); 
      trainTime = cputime - startTime;                     
    end
  end % methods(Static)
end % class

