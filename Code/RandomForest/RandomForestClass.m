classdef RandomForestClass < handle
  %RANDOMFORESTCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    hyperparameters;
    debug = false;
  end
  
  methods
    function obj = RandomForestClass(rfHyperparameters)
      obj.hyperparameters = rfHyperparameters;
    end
    
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
      randomTreeBagSeed = 1;
      for numFeature = obj.hyperparameters.features
        for numTree = obj.hyperparameters.trees
          for numFolds = obj.hyperparameters.folds
            errTrain = zeros(size(numFolds,2),1);
            errValid = zeros(size(numFolds,2),1);
            errOob = zeros(size(numFolds,2),1);
            accuracies = zeros(size(numFolds,2),1);
            precisions = zeros(size(numFolds,2),1);
            recalls = zeros(size(numFolds,2),1);
            f1s = zeros(size(numFolds,2),1);
            startTime = cputime;
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
              %{
              model = TreeBagger(numTree, xTrain, yTrain, 'OOBPrediction','On',...
                                'Method','classification', ...
                                'Prior', priorDistribution, ...
                                'NumPredictorsToSample', numFeature);
                            errTrain(foldCount) = mean(error(model, xTrain, yTrain));
              errValid(foldCount) = mean(error(model, xTest, yTest));
              errOob (foldCount)  = mean(oobError(model));
              %}
              [trainingErr, testErr, oobErr, misclassifiedCount, ...
                    model, accuracy, precision, recall, f1] = ...
                   obj.buildAndTestTreeBagger(numTree, xTrain, yTrain, xTest, yTest, ...
                                priorDistribution, numFeature, randomTreeBagSeed);              
              errTrain(foldCount) = trainingErr;
              errValid(foldCount) = testErr;
              errOob(foldCount)  = oobErr
              accuracies(foldCount) = accuracy
              precisions(foldCount) = precision;
              recalls(foldCount) = recall;
              f1s(foldCount) = f1;
              if obj.debug
                % check cross valid is different
                fprintf("tree %d feature %d trainError %0.04f test error %0.04f "+ ...
                        "oob error %0.04f fold %d  of %d " + ...
                        "\tSamples: %s %s %s %s %s\n", ...
                        numTree, numFeature, errTrain(foldCount), errValid(foldCount), ...
                        errOob(foldCount),foldCount, numFolds, ...
                        string(table2array(yTrain(1,1))), ...
                        string(table2array(yTrain(2,1))), ...
                        string(table2array(yTrain(3,1))), ...
                        string(table2array(yTrain(4,1))), ...
                        string(table2array(yTrain(5,1))));
              end %debug
            end %foldCount
            elapsedTime = cputime - startTime;
            rfResults.appendResult(numFolds, numTree, numFeature, randomTreeBagSeed, mean(errTrain), ...
              mean(errValid), mean(errOob), ...
              mean(accuracies), mean(precisions), mean(recalls), mean(f1s), ...
              size(yTest, 1), elapsedTime);
            randomTreeBagSeed = randomTreeBagSeed + 1;
          end % folds
        end % tree
      end % feature
      rfResults.endGatheringResults();
    end % function
    
    function [trainingLoss, testLoss, oobErr, misclassifiedCount, ...
                    model, accuracy, precision, recall, f1] = ...
              buildAndTestTreeBagger(obj, numTree, xTrain, yTrain, xTest, yTest, ...
                                    priorDistribution, numFeature, randomSeed)
        rng(randomSeed);
        model = TreeBagger(numTree, xTrain, yTrain, 'OOBPrediction','On',...
                                'Method','classification', ...
                                'Prior', priorDistribution, ...
                                'NumPredictorsToSample', numFeature); 
        trainingLoss = mean(error(model, xTrain, yTrain));                              
        testLoss = mean(error(model, xTest, yTest));
        oobErr = mean(oobError(model));
      % predict using given xTest
      [predictionResult, ~] = predict(model, xTest);
      % errors
      misclassifiedCount = sum(predictionResult ~= categorical(table2array(yTest)));

      [accuracy, precision, recall, ...
        f1] = CalcUtil.calculateMeasuresFromExpectPredict(table2array(yTest), predictionResult);
      if obj.debug
        % display random 5 predictions, check loss and misclassified
        % percent match
        %predictionResult(randsample(numel(predictionResult), 5))
        percentMisclassified = (misclassifiedCount / size(yTest, 1)) * 100;
        fprintf("Misclassified %d entries out of %d. %0.04f pct\n", misclassifiedCount, size(yTest, 1), percentMisclassified);
        fprintf("\tTraining loss: %0.02f. Test Loss: %0.02f Oob Error: %0.02f\n", trainingLoss, testLoss, oobErr);
        fprintf("\tPrecision: %0.04f Recall: %0.04f Accuracy: %0.04f F1: %0.04f\n", ...
                 precision, recall, accuracy, f1);      
      end % debug        
    end % function
    
  end % methods
end % class

