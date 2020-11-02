%%
% Decision tree class
classdef LetterDecisionTreeClass < handle
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
    function treeModel = buildTree(obj)
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
    
    function trainTree(obj)
      [obj.trainedClassifier, obj.validationAccuracy] = vanillaTrainClassifier(obj.dataset.trainData);
    end
  end
  
  
end % class