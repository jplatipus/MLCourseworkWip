classdef NBayesHyperparameterClass < handle
  %NBAYESHYPERPARAMETERCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
      randomSeed = 110;
      % distribution names used in the default (normal) distribution
      distNamesDefault = {'normal','normal','normal','normal','normal','normal','normal','normal','normal',...
        'normal','normal','normal','normal','normal','normal','normal'};
      % distribution names used in the kernel distribution
      distNamesKernel = {'kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel',...
      'kernel','kernel','kernel','kernel','kernel','kernel','kernel'};
      numberOfHoldOutRuns = [1 5 10 15 20];
      distributionNames = [distNamesDefault, distNamesKernel];
      kernelWidths = [];
  end % properties
  
  methods
    function obj = NBayesHyperparameterClass()

    end
  end % methods
    
  methods(Static)
    function hyperparameters = getQuickTestHyperparametersInstance()
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [2];
      hyperparameters.distributionNames = [distNamesDefault];
      hyperparameters.kernelWidths = [0.25916];
    end
    
  end % methods(static)
end % class

