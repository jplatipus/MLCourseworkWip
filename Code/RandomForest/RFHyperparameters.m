classdef RFHyperparameters < handle
  % Class stores the hyperparameter values that are used in the
  % hyperparameter search.
  
  properties
    features = [2, 4, 6, 8, 10, 12];
    trees = [50, 100, 150, 200];
    folds = [2, 3, 5];
    randomSeed = 122;
  end % properties
  
  methods
    function obj = RFHyperparameters()
    end % constructor
  end % methods
  
 methods(Static)
   % create an instance for testing the code: few hyperparameters to try
   function rfHyperparameters = getHyperQuickInstance()
     rfHyperparameters = RFHyperparameters();
     rfHyperparameters.features = [2, 4];
     rfHyperparameters.trees = [25];
     rfHyperparameters.folds = [2, 3]
   end % function
  
   % get the hyperparameter values that are used in the search
   function rfHyperparameters = getHyperDefaultInstance()
     rfHyperparameters = RFHyperparameters();
   end % function  
 end % methods(Static)
end % class

