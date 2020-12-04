classdef FinalRFModel < handle
  %FINALRFMODEL Summary of this class goes here
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
    function buildFinalModel(obj, xTrain, yTrain, priorDistribution, numberOfTrees, numberOfFeatures, randomSeed)
      rng(randomSeed);
      startTime = cputime;
      obj.model = TreeBagger(numberOfTrees, xTrain, yTrain, 'OOBPrediction','On',...
                              'Method','classification', ...
                              'Prior', priorDistribution, ...
                              'NumPredictorsToSample', numberOfFeatures); 
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

