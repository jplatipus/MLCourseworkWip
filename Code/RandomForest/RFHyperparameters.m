classdef RFHyperparameters < handle
  %RFHYPERPARAMETERS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    features = [2, 4, 6, 8, 10, 12];
    trees = [50, 100, 150, 200];
    folds = [2, 3, 5];
    randomSeed = 122;
  end % properties
  
  methods
    function obj = RFHyperparameters()
    end % constructor
    
    function outputArg = method1(obj,inputArg)
      %METHOD1 Summary of this method goes here
      %   Detailed explanation goes here
      outputArg = obj.Property1 + inputArg;
    end % function
  end % methods
  
 methods(Static)
   function rfHyperparameters = getHyperQuickInstance()
     rfHyperparameters = RFHyperparameters();
     rfHyperparameters.features = [2, 4];
     rfHyperparameters.trees = [25];
     rfHyperparameters.folds = [2, 3]
   end % function
  
   function rfHyperparameters = getHyperDefaultInstance()
     rfHyperparameters = RFHyperparameters();
   end % function
   
   % the hyperparameters to use for the final model are: 
   % 200 trees, 
   % 4 features, 
   % 60 as the random seed.
   function rfHyperparameters = getHyperparametersFinalInstance()
     rfHyperparameters = RFHyperparameters();
     rfHyperparameters.trees = 200;
     rfHyperparameters.features = 4;
     rfHyperparameters.randomSeed = 60;
   end
   
 end % methods(Static)
end % class

