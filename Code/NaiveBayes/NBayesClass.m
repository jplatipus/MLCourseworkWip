classdef NBayesClass < handle
  %NBAYESCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    randomSeed = 300;
    debug = true;
    dataset;
    nBayesModel;
    x;
    y; 
    xt; 
    yt;
    distNamesDefault = {'normal','normal','normal','normal','normal','normal','normal','normal','normal',...
      'normal','normal','normal','normal','normal','normal','normal'};
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
    
    function fitModel(obj, distributionNames)
        obj.nBayesModel = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'DistributionNames', distributionNames);
    end % method
    
    function [trainingLoss, testLoss] = getModelLoss(obj)
        trainingLoss = loss(obj.nBayesModel, obj.x, obj.y);
        testLoss = loss(obj.nBayesModel, obj.xt, obj.yt);
    end
    
    function nBayesModel = setPriorDistributionEmpirical(obj)
      Y = table2array(obj.y);
      freqDist = cell2table(tabulate(Y));
      prior = freqDist{:,3};
      obj.nBayesModel.Prior = prior;
      nBayesModel = obj.nBayesModel;
    end
    
    function nBayesModel = fitMatlabHyperparameterOptimization(obj)
      obj.nBayesModel = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus'))
      nBayesModel = obj.nBayesModel;
    end

  end %methods
  
  %
  % Static methods
  %
  methods(Static)

    function nBayesClassInstance = getDefaultInstance(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
      nBayesClassInstance.fitModel(nBayesClassInstance.distNamesDefault);
    end %function    
    
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

