%
% DatasetClass wraps all the dataset load and processing operations
%
classdef DatasetClass < handle
    properties  
      % Path to dataset file that is loaded:
      datasetFilePath = "Dataset/letter-recognition.csv";
      % Proportion of dataset to load in the test set (1 - proportion for
      % training and validation):
      testSetProportion = 0.2;
      
      trainData = {};
      testData = {};
    end % properties
    
    methods
        
        % Public
        % Constructor: load dataset, split data into class members (tables)
        %
        function obj = DatasetClass()
          %Load file: 
          datasetContentsAsTable = readtable(obj.datasetFilePath, 'Delimiter', ',');

          % split the dataset into train/validate and test
          exampleCount = size(datasetContentsAsTable, 1);
          trainValidateExampleCount = round(exampleCount *  (1 - obj.testSetProportion));
          % dataset is already shuffled, so we can take sequential examples
          % for train and test:
          obj.trainData = datasetContentsAsTable(1:trainValidateExampleCount, :);
          obj.testData = datasetContentsAsTable(trainValidateExampleCount + 1:end, :);
          
          % Assign variable names to the tables
          featureNames = {'TargetAscii', 'xBox', 'yBox', 'width', 'height', 'onPixelCount', 'xBar', 'yBar', 'x2Bar', 'y2Bar', 'xyBar', 'x2yBr', 'xy2Br', 'xEge', 'xEgvy', 'yEge', 'yEgvx'};
          obj.trainData.Properties.VariableNames = featureNames
          obj.testData.Properties.VariableNames = featureNames
          
        end
        
    end % methods
    
    %
    % Static methods
    %
    methods(Static)
      
    end % static methods
    
end % class