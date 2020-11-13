classdef NBayesClass < handle
  %NBAYESCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    randomSeed = 300;
    debug = true;
    dataset;
    distNames = {'kernel','mvmn','mvmn','mvmn','mvmn','kernel','mvmn','mvmn','mvmn',...
    'kernel','mvmn','kernel','kernel','kernel','kernel','mvmn'};
  end
  
  methods
    function obj = NBayesClass(letterDataset)
      obj.dataset = letterDataset;
    end
    
    function nBayesModel = simpleNaiveBayesClassifier(obj)
      rng(obj.randomSeed);
      [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      [xt, yt] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
      nBayesModel = fitcnb(x, y, 'ClassNames', categorical(table2array(obj.dataset.validClassValues)));

      if obj.debug
        trainingLoss = loss(nBayesModel, x, y);
        testLoss = loss(nBayesModel, xt, yt);
        fprintf("Training loss: %0.02f Test loss: %0.02f\n", trainingLoss, testLoss);
      end
    end %function
    
    function nBayesModel = kernelClassifier(obj)
      
    end % function
  end %methods
end

