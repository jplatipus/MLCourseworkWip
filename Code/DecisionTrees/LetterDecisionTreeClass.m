classdef LetterDecisionTreeClass < handle
%%
% Decision tree class
  properties
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
         
    % NO Folds in hold out. Simpler than kfold crossvalidation
    % Good for large datasets: we have
    % a largish one at 16000 examples for the train/validate, and 16
    % features.
    function performDTreeHyperameterAnalysis(obj, hyperparameters)
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      for trainValidateProportion = hyperparameters.trainValidateProportions
        for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns
          % repeat holdout partition creation, build tree, predict this
          % number of times to get average:
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
            for maxNumSplit = hyperparameters.maxNumSplits
              for splitCriterion = hyperparameters.splitCriteria
                obj.buildAndTestTree(xTrain, yTrain, xTest, yTest, ...
                                     splitCriterion, maxNumSplit, classNames);
              end
            end
          end
        end
      end 
    end % function

    %
    % Build a decision tree, and test it using the passed parameters 
    % fitctree, these parameters are the values passed directly to fitctree:
    %   xTrain, yTrain, splitCriterion, maxNumSplit, classNames
    % predict:
    %   xTest is passed to predict
    %   yTest are the expected results.
    %
    function buildAndTestTree(obj, xTrain, yTrain, xTest, yTest, splitCriterion, maxNumSplit, classNames)
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
      % display several prediction results
      predictionResult(randsample(numel(predictionResult), 5))
      % errors
      trainingLoss = loss(treeModel, xTrain, yTrain);
      testLoss = loss(treeModel, xTest, yTest);
      fprintf("Training loss: %0.02f. Test Loss: %0.02f\n", trainingLoss, testLoss);
    end % function
    
  end % methods
  
  
end % class