classdef LetterDatasetClass < handle
    properties  
      % Path to dataset file that is loaded:
      datasetFilePath = "Dataset/letter-recognition.csv";
      % Proportion of dataset to load in the test set (1 - proportion for
      % training and validation):
      testSetProportion = 0.2;
      % Random number generator initial seed for shuffling the dataset:
      randomSeed = 1;
      % dataset feature names:
      targetName = {'TargetAscii'};
      featureNames = {'xBox', 'yBox', 'width', 'height', 'onPixelCount', 'xBar', 'yBar', 'x2Bar', 'y2Bar', 'xyBar', 'x2yBr', 'xy2Br', 'xEge', 'xEgvy', 'yEge', 'yEgvx'};
      validClassValues = {};
      allData = {};
      trainData = {}; 
      xTrain = {};
      yTrain = [];
      testData = {};
      xTest = {};
      yTest = {};
    end % properties
    
    methods
        
        % Public
        % Constructor: load dataset, split data into class members (tables)
        %
        function obj = LetterDatasetClass()
          %Load file: 
          datasetContentsAsTable = readtable(obj.datasetFilePath, 'Delimiter', ',');
          %Normalise dataset

          
          %Shuffle dataset
          rng(obj.randomSeed)
          randomRowIndices = randperm(size(datasetContentsAsTable,1));
          datasetContentsAsTable = datasetContentsAsTable(randomRowIndices, :);
          % normalise dataset
          datasetContentsAsTable = normalize(datasetContentsAsTable, 'norm', Inf, 'DataVariables', @isnumeric);
          obj.allData = datasetContentsAsTable;

          % split the dataset into train/validate and test
          exampleCount = size(datasetContentsAsTable, 1);
          trainValidateExampleCount = round(exampleCount *  (1 - obj.testSetProportion));
          % dataset is already shuffled, so we can take sequential examples
          % for train and test:
          obj.trainData = datasetContentsAsTable(1:trainValidateExampleCount, :);
          obj.testData = datasetContentsAsTable(trainValidateExampleCount + 1:end, :);          
          %Extract set of class values
          obj.validClassValues = unique(datasetContentsAsTable(:,1));
          % set properties:
          obj.trainData.Properties.VariableNames = [obj.targetName, obj.featureNames];
          obj.xTrain = obj.trainData(:,2:end);
          obj.yTrain = obj.trainData(:,1);
          obj.testData.Properties.VariableNames = [obj.targetName, obj.featureNames];;
          obj.xTest = obj.testData(:,2:end);
          obj.yTest = obj.testData(:,1);
        end
        
        %
        % Display correlation of training data as a heatmap
        %
        function displayCorrelation(obj)
          figure("Name", "Correlation");
          mat = table2array(obj.xTrain);
          covariance = corrcoef(mat);       
          h = heatmap(obj.featureNames,obj.featureNames,covariance);
          h.title("Attribute Correlation Heatmap");
        end
        
        %
        % Display a scatter plot matrix of of the attributes (processor
        % intensive)
        %
        function displayScatterMatrix(obj)
          figure("Name", "Scatter Matrix of Attributes");
          mat = table2array(obj.xTrain);
          targetMat = table2array(obj.yTrain);
          xnames = obj.featureNames;
          ynames = xnames;
          color = lines(26);
          gplotmatrix(mat, mat, targetMat, color, [], [], [], 'variable', xnames, ynames);
          title("Scatter Matrix of Attributes");
        end
        
    end % methods
    
    %
    % Static methods
    %
    methods(Static)
      
    end % static methods
    
end % class