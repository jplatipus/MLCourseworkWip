classdef NBayesHyperparameterClass < handle
  %NBAYESHYPERPARAMETERCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
      randomSeed = 110;
      numberOfFolds = [1 5 10 15 20];
      distributionNames = [];
      kernelSmoother = [];
      kernelWidths = [];
  end % properties
  
  methods
    function obj = NBayesHyperparameterClass()

    end
  end % methods
    
  methods(Static)
    %% Convenience hyperparameter constructors (static)
    
    % get an instance for testing the code works, without kernel widths: 
    % quick run through most of the code
    function hyperparameters = getQuickTestHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2];
      % rows of distribution names (eg: kernel, kernel, ....)
      hyperparameters.distributionNames = {[nBayes.distNamesDefault]};
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [-1.0];
    end
    
    % get an instance for testing the code works, with kernel widths: 
    % quick run through most of the code
    function hyperparameters = getKernalNormalQuickHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = { nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothNormal, nBayes.smoothNormal, ...
        nBayes.smoothNormal};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.20 0.18 0.13];
    end
    
    % get kernel box smoother instance
    function hyperparameters = getKernalBoxSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2 3 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = {nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothBox, nBayes.smoothBox, ...
        nBayes.smoothBox, nBayes.smoothBox, nBayes.smoothBox, ...
        nBayes.smoothBox, nBayes.smoothBox, nBayes.smoothBox};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.20 0.18 0.13 0.12 0.11 0.10 0.090 0.80];
    end
    
    % get Epanechnikov smoother instance
    function hyperparameters = getKernalEpanechnikovSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2 3 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = {nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov, ...
        nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov, ...
        nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.20 0.18 0.13 0.12 0.11 0.10 0.090 0.80];
    end
    
    % get normal (Gaussian) smoother instance
    function hyperparameters = getKernalNormalSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2 3 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = { nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...                                        nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothNormal, nBayes.smoothNormal, ...
        nBayes.smoothNormal, nBayes.smoothNormal, nBayes.smoothNormal, ...
        nBayes.smoothNormal, nBayes.smoothNormal, nBayes.smoothNormal};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.20 0.18 0.13 0.12 0.11 0.10 0.090 0.80];
    end
    
    % instance to see how wild kernel widths affect the predictions for a
    % normal kernel smoother
    function hyperparameters = getKernalNormalSmootherError(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = { nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            ...
                                            nBayes.distNamesKernel, ...                                        nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {
        nBayes.smoothNormal, nBayes.smoothNormal, nBayes.smoothNormal, ...
        nBayes.smoothNormal, nBayes.smoothNormal, nBayes.smoothNormal, ...
        nBayes.smoothNormal, nBayes.smoothNormal, nBayes.smoothNormal};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [2.00 1.00 0.10 ...
                                      0.01 0.001 0.0001 ...
                                      0.0000001 0.00000001 0.000000001];
    end
        
    % get triangle kernel smoother instance
    function hyperparameters = getKernelTriangleSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2 3 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = {nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothTriangle, nBayes.smoothTriangle, ...
        nBayes.smoothTriangle, nBayes.smoothTriangle, nBayes.smoothTriangle, ...
        nBayes.smoothTriangle, nBayes.smoothTriangle, nBayes.smoothTriangle};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.20 0.18 0.13 0.12 0.11 0.10 0.090 0.80];
    end
    
    % get MATLAB's default 'normal' instance. Widths are not used
    function hyperparameters = getDefaultTestHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfFolds = [2 3 5];
      % rows of distribution names (eg: kernel, kernel, ....)
      hyperparameters.distributionNames = {nBayes.distNamesDefault };
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [-1.0];
      hyperparameters.kernelSmoother = {nBayes.smoothUnsused};
    end
    
  end % methods(static)
end % class

