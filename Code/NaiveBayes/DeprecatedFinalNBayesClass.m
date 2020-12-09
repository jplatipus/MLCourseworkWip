classdef FinalNBayesClass < handle
  %FINALNBAYESMODEL Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    model;
    modelCreationTime;
    modelPredictTime;
    misclassifiedCount;
    accuracy;
    precision;
    recall;
  end
  
  methods
    
    % Train the final model given the train hyperparameters.
    % save model and time in class member variables
    function buildFinalModel(obj, xTrain, yTrain, priorDistribution, ...
        nBayesRandomSeed, ...
        classNames, distributionNames, smootherTypes, width)
      rng(nBayesRandomSeed);
      startTime = cputime;
      % train naive bayes model
      obj.model = fitcnb(xTrain, yTrain, ...
        'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
        'Prior', priorDistribution, 'kernel', string(smootherTypes{:,:}), 'width', width);
      endTime = cputime;
      obj.modelCreationTime = endTime - startTime;                  
    end
    
    % perform predict on the model using the given test values
    % save the calculated metrics and time in the class
    function peformPredict(obj, xTest, yTest)
      startTime = cputime;
      [predictionResult, ~] = predict(obj.model, xTest);
      endTime = cputime;
      obj.modelPredictTime = endTime - startTime;
      obj.misclassifiedCount = sum(predictionResult ~= categorical(table2array(yTest)));

      [obj.accuracy, obj.precision, obj.recall, ...
        obj.f1] = CalcUtil.calculateMeasuresFromExpectPredict(table2array(yTest), predictionResult);
    end
  end
end

