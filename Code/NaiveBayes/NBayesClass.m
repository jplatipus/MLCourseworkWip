classdef NBayesClass < handle
  %% NBayesClass
  % contains the code to perform the initial automatic hyperparameter
  % search, the detailed hyperparameter search, and for creating the final
  % model.
  properties
    % default random stream initial value
    randomSeed = 300;
    % random stream used by default
    defaultRandomStream;
    % random stream used when training naive bayes
    nBayesRandomStream;
    debug = false;
    % letter dataset used in this instance:
    dataset;
    % Matlab Model:
    nBayesModel;
    % Training attributes:
    x;
    % Training results:
    y; 
    % test attributes:
    xt; 
    % test results:
    yt;
   
    % distribution names used in the default (normal) distribution
    distNamesDefault = {'normal','normal','normal','normal','normal','normal','normal','normal','normal',...
      'normal','normal','normal','normal','normal','normal','normal'};
    % distribution names used in the kernel distribution
    distNamesKernel = {'kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel',...
      'kernel','kernel','kernel','kernel','kernel','kernel','kernel'};
    %
    % kernel smooting parameters to use for each feature.
    %
    smoothBox = {'box', 'box', 'box', 'box', 'box', 'box', 'box', 'box', 'box', ...
      'box', 'box', 'box', 'box', 'box', 'box', 'box'};
    smoothEpanechnikov = { 'epanechnikov', 'epanechnikov', 'epanechnikov', ...
      'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov', ...
      'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov', ...
      'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov'};
    smoothNormal = { 'normal', 'normal', 'normal', 'normal', 'normal', ...
      'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', ...
      'normal', 'normal', 'normal', 'normal'};
    smoothTriangle = {'triangle', 'triangle', 'triangle', 'triangle', 'triangle',...
      'triangle', 'triangle', 'triangle', 'triangle', 'triangle', 'triangle', ...
      'triangle', 'triangle', 'triangle', 'triangle', 'triangle'};
    smoothUnsused = {'N/A', 'N/A', 'N/A', 'N/A', 'N/A',...
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', ...
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A'};
  end % Properties

  methods
    
    %% Constructor
    % Several getInstance static convenience methods use this constructor.
    % Construct an instance with the given dataset. Creates two random
    % streams: defaultRandomStream for partitioning the training data,
    % nBayesRandomStream is used before fitting the model to the Naive Bayes method.
    function obj = NBayesClass(letterDataset)
      obj.dataset = letterDataset;
      obj.defaultRandomStream = RandStream('mt19937ar');
      obj.nBayesRandomStream = RandStream('mt19937ar');
      RandStream.setGlobalStream(obj.defaultRandomStream)      
      rng(obj.randomSeed);
      [obj.x, obj.y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      [obj.xt, obj.yt] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
    end % constructor
    
    % Fit the model, set nBayesModel to the model
    function fitModel(obj, distributionNames)
        model = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'DistributionNames', distributionNames);
      obj.nBayesModel = model;
    end % method
    
    % calculate the model's training and test loss
    function [trainingLoss, testLoss] = getModelLoss(obj)
        trainingLoss = loss(obj.nBayesModel, obj.x, obj.y);
        testLoss = loss(obj.nBayesModel, obj.xt, obj.yt);
    end
    
    % Set the Matlab model's empirical prior distribution to reflect the dataset's
    % training distribution of classes.
    % Update nBayesModel and return it too
    function nBayesModel = setPriorDistributionEmpirical(obj)
      prior = obj.getPriorDistributionEmpirical();
      obj.nBayesModel.Prior = prior;
      nBayesModel = obj.nBayesModel;
    end % function
    
    % calculate the dataset's
    % training empirical distribution of classes
    % returns a vector of the distributions for each class
    function prior = getPriorDistributionEmpirical(obj)
      Y = table2array(obj.y);
      freqDist = cell2table(tabulate(Y));
      prior = freqDist{:,3}/100;
    end
       
    % Perform matlab's hyperparameter optimization search.
    % sets the prior distribution to empirical
    % returns the MATLAB model (also set in obj.nBayesModel)
    function nBayesModel = fitMatlabHyperparameterOptimization(obj)
      obj.nBayesModel = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus'), ...
        'Prior', obj.getPriorDistributionEmpirical());
      nBayesModel = obj.nBayesModel;
    end % function

    %
    % Perform a 'manual' nfold cross validation with replacement hyperparameter search,
    % using the different hyperparameter values supplied in
    % hyperparameters.
    % writes the results to the csv file (tab delimitted), and returns the
    % results as a results class instance.
    %
    function letterNBayesResults = performHyperparameterSearch(obj, hyperparameters, resultsCsvFilename)
      fprintf("Starting Naive Bayes analysis...\n");
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = NBayesResults(resultsCsvFilename);
      resultsTable.startGatheringResults();  
      % for gaussian kernel distribution, we can specify a prior
      % distribution of classes. This test uses emprirical
      % (based on the training set distribution):
      priorDistributions = obj.getPriorDistributionEmpirical(); %, ...
                      % obj.getPriorDistributionNormal()];
      cellsIndex = 1;
      resultRow = 1;
      % iterate over the hyperparameter values
      for distributionNameCells = hyperparameters.distributionNames 
        for numberOfFolds = hyperparameters.numberOfFolds       
          for priorDistribution = priorDistributions
              % repeat fold partition creation, build tree, predict this
              startTime = cputime; 
              trainLosses = zeros(1, numberOfFolds);
              testLosses = zeros(1, numberOfFolds);
              misclassificationCounts = zeros(1, numberOfFolds);  
              accuracies = zeros(1, numberOfFolds);
              precisions = zeros(1, numberOfFolds);
              recalls = zeros(1, numberOfFolds);
              f1s = zeros(1, numberOfFolds);
              predictTimes = zeros(1, numberOfFolds);
              % for each fold
              for foldCount = 1:numberOfFolds
                % get width row matching distributionNames row
                width = hyperparameters.kernelWidths(cellsIndex);
                % get smoother type matching distribution names row
                smootherType = hyperparameters.kernelSmoother(cellsIndex);
                % calculate train and test subset indeces
                partition = cvpartition(numExamples, 'Holdout', 0.2);
                trainSubsetIdx = training(partition);
                testSubsetIdx = test(partition);
                [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
                % extract train and test subsets
                xTrain = x(trainSubsetIdx, :);
                yTrain = y(trainSubsetIdx, :);
                xTest = x(testSubsetIdx, :);
                yTest = y(testSubsetIdx, :);
                if obj.debug
                  fprintf("Test run fold: %d of %d,", foldCount,numberOfFolds);
                  fprintf("Width: %0.04f ",width);
                  fprintf("\n\tSmoother Type: ");
                  for str = string(smootherType{1,1})
                    fprintf("%s ", str);
                  end
                  fprintf("\n\tDistName: ");
                  for str = string(distributionNameCells{1,1})
                    fprintf("'%s' ", str);
                  end
                  fprintf("\n\tprior: "); 
                  for aPrior = priorDistribution
                    fprintf("%0.04f ", aPrior);
                  end
                  fprintf('\n');
                end %obj.debug;
                % uses the result row as the random seed, so that each fold
                % uses the same random seed when fitting the model.
                [trainingLoss, testLoss, misclassifiedCount, ...
                    distributionName, smootherTypeName, model, ...
                    accuracy, precision, recall, ...
                    f1, predictTime] =...
                  obj.buildAndTestNBayes(xTrain, yTrain, xTest, yTest, ...
                    distributionNameCells, smootherType, priorDistribution, width, ...
                    classNames, resultRow);
                  % gather results for this fold
                accuracies(foldCount) = accuracy;
                precisions(foldCount) = precision;
                recalls(foldCount) = recall;
                f1s(foldCount) = f1;
                trainLosses(foldCount) = trainingLoss;
                testLosses(foldCount) = testLoss;
                predictTimes(foldCount) = predictTime;
                misclassificationCounts(foldCount) = misclassifiedCount;
              end % foldCount
              endTime = cputime;
              avgTrainLoss = mean(trainLosses);
              avgTestLoss = mean(testLosses);
              avgAccuracy = mean(accuracies);
              avgPrecision = mean(precisions);
              avgRecall = mean(recalls);
              avgF1 = mean(f1s);
              avgPredictTime = mean(predictTimes);
              % Uses the result row number as the random seed when training
              % the naive bayes model.
              resultsTable.appendResult(distributionName, smootherTypeName, ...
                                 width, numberOfFolds, resultRow, ...
                                 avgTrainLoss, avgTestLoss, ...
                                 avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                                 size(yTest, 1), ...
                                 endTime - startTime, avgPredictTime);
              resultRow = resultRow + 1;
          end % priorDistribution
        end % numberOfFolds  
        cellsIndex = cellsIndex + 1;
      end % distributionName   
      resultsTable.endGatheringResults();
      letterNBayesResults = resultsTable;
      fprintf("Completed NBayes analysis\n");      
    end % function
    
    % Build the model using the training set, predict using the test set,
    % using the hyperparameters supplied.
    % calculate and return:
    % trainingLoss, testLoss, misclassifiedCount, ...
    %   distributionName, model, ...
    %    classAccuracy, classPrecision, classRecall, ...
    %    classF1, classLabel
    function [trainingLoss, testLoss, misclassifiedCount, ...
        distributionName, smootherTypeName,model, ...
        accuracy, precision, recall, ...
        f1, predictTime] = buildAndTestNBayes(obj, xTrain,  yTrain, xTest, yTest, ...
                        distributionNames, smootherTypes, ...
                        priorDistribution, width, classNames, nBayesRandomSeed)
      % train naive bayes model
      % use the random number generator instance for growing the ensemble
      RandStream.setGlobalStream(obj.nBayesRandomStream);
      rng(nBayesRandomSeed);
      model = [];
      if width >= 0.0
        % a width >= 0 means that the distribution name is 'kernel', and
        % smoother types are used
        model = fitcnb(xTrain, yTrain, ...
          'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
          'Prior', priorDistribution, 'kernel', string(smootherTypes{:,:}), 'width', width);
        distributionName = 'kernel';
        cellConverter = smootherTypes{1,1};
        smootherTypeName = cellConverter{1,1};
      else
        % a width < 0 means that there is no width parameter associated with
        % the distribution names (normal) and no smoother types        
        model = fitcnb(xTrain, yTrain, ...
          'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
          'Prior', priorDistribution);
        distributionName = 'normal';
        smootherTypeName = 'unused';
      end
      % restore the random number generator instance to the "default" one
      RandStream.setGlobalStream(obj.defaultRandomStream);
      % predict using given xTest
      startTime = cputime;
      [predictionResult, ~] = predict(model, xTest);
      predictTime = cputime - startTime;
      % errors
      misclassifiedCount = sum(predictionResult ~= categorical(table2array(yTest)));
      trainingLoss = loss(model, xTrain, yTrain);
      testLoss = loss(model, xTest, yTest);
      [accuracy, precision, recall, ...
        f1] = CalcUtil.calculateMeasuresFromExpectPredict(table2array(yTest), predictionResult);
      if obj.debug
        % display random 5 predictions, check loss and misclassified
        % percent match
        %predictionResult(randsample(numel(predictionResult), 5))
        percentMisclassified = (misclassifiedCount / size(yTest, 1)) * 100;
        fprintf("Misclassified %d entries out of %d. %0.04f pct\n", misclassifiedCount, size(yTest, 1), percentMisclassified);
        fprintf("\tTraining loss: %0.02f. Test Loss: %0.02f\n", trainingLoss, testLoss);
        fprintf("\tPrecision: %0.04f Recall: %0.04f Accuracy: %0.04f F1: %0.04f\n", ...
                 precision, recall, accuracy, f1);      
      end % debug
    end % function
    
  end %methods
  
  %
  % Static methods
  %
  methods(Static)

    % Get default, unfitted class instance
    function nBayesClassInstance = getDefaultInstanceNoFit(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
    end %function 
    
    % Get the default class instance for the given dataset
    function nBayesClassInstance = getDefaultInstance(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
      nBayesClassInstance.fitModel(nBayesClassInstance.distNamesDefault);
    end %function    
    
    % Get the kernel distribution class instance for the given dataset
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
      nBayesClassInstance.smoothBox = nBayesClassInstance.smoothBox(:,1:attributeCount - 1);
      nBayesClassInstance.smoothEpanechnikov = nBayesClassInstance.smoothEpanechnikov(:,1:attributeCount - 1);
      nBayesClassInstance.smoothNormal = nBayesClassInstance.smoothNormal(:,1:attributeCount - 1);
      nBayesClassInstance.smoothTriangle = nBayesClassInstance.smoothTriangle(:,1:attributeCount - 1);
    end
    
    % build final Naive Bayes model
    % returns the trained model and the cpu time taken to do this.
    function [nBayes, trainTime] = buildFinalNBayes(distributionNames, ...
        smootherTypes, width, nBayesRandomSeed, classNames, xTrain, yTrain)
      % calculate prior distribution of classes based on training dataset
      yAsArray = table2array(yTrain);
      freqDist = cell2table(tabulate(yAsArray));
      priorDistribution = freqDist{:,3}/100;
      rng(nBayesRandomSeed);
      startTime = cputime;
      % train naive bayes model
      nBayes = fitcnb(xTrain, yTrain, ...
        'ClassNames', classNames, 'DistributionNames', [distributionNames{:,:}], ...
        'Prior', priorDistribution, 'kernel', [smootherTypes{:,:}], 'width', width);
      endTime = cputime;
      trainTime = endTime - startTime;                   
    end
  end % static methods
end

