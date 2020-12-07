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
      % Unique set of class values extracted from the dataset:
      validClassValues = {};
      % Complete normalised dataset:
      datasetContentsAsTable = {};
      % random training examples:
      trainTable = {};
      % random test examples:
      testTable = {};
      % flag indicating if the dataset has been normalised: 
      % (Center and scale to have mean 0 and standard deviation 1)
      isNormalised = false;
      % has some features been removed
      isRemovedFeature = false;
    end % properties
    
    methods
        
        % Public
        % Constructor: load dataset, split data into class members (tables)
        %
        function obj = LetterDatasetClass(normalised)
          %Load file: 
          obj.datasetContentsAsTable = readtable(obj.datasetFilePath, 'Delimiter', ',');
          
          %Extract set of class values
          obj.validClassValues = unique(obj.datasetContentsAsTable(:,1));
          %Split dataset into train and test sets
          rng(obj.randomSeed);
          exampleCount = size(obj.datasetContentsAsTable, 1);
          partition = cvpartition(exampleCount, 'Holdout', obj.testSetProportion);
          idxTrain = training(partition);
          idxTest = test(partition);
          obj.trainTable = obj.datasetContentsAsTable(idxTrain, :);
          obj.testTable = obj.datasetContentsAsTable(idxTest, :);
          % set table column names:
          obj.trainTable.Properties.VariableNames = [obj.targetName, obj.featureNames];
          obj.testTable.Properties.VariableNames = [obj.targetName, obj.featureNames];
          obj.datasetContentsAsTable.Properties.VariableNames = [obj.targetName, obj.featureNames];
          % normalise the data using zscore standardization?
          if normalised
            obj.performStandardization();
          end
        end
        
        %
        % Feature selection: removes the given column from the datasetContentsAsTable,
        % train, test tables and the featureNames.
        %
        function removeColumn(obj, columnName)
          obj.featureNames = obj.featureNames(:,~strcmp(obj.featureNames, columnName));
          obj.trainTable = removevars(obj.trainTable, columnName);
          obj.testTable = removevars(obj.testTable, columnName);
          obj.datasetContentsAsTable = removevars(obj.datasetContentsAsTable, columnName);
          obj.isRemovedFeature = true;
        end
        
        %
        % Normalizes the dataset trainTable and testTable. The test table
        % is normalised using the train table so as to avoid leakage into
        % the test set.
        % The member isNormalised reflects whether the train table and
        % test table have been normalised
        % algorithm used in the normalization is:
        % - zscore normalization
        % On exit the members trainTable and testTable have been
        % normalised
        function performStandardization(obj)
          % claculate training mean and std dev
          trainingMeans = mean(table2array(obj.trainTable(:,2:end)));
          trainingStdDevs = std(table2array(obj.trainTable(:,2:end)));
          % normalise training table
          zvalues = (table2array(obj.trainTable(:,2:end)) - trainingMeans) ./ trainingStdDevs;
          zTable = obj.trainTable;
          zTable(:,2:end) = array2table(zvalues);
          obj.trainTable = zTable;        
          % normalise test table
          zvalues = (table2array(obj.testTable(:,2:end)) - trainingMeans) ./ trainingStdDevs;
          zTable = obj.testTable;
          zTable(:,2:end) = array2table(zvalues);
          obj.testTable = zTable;
          obj.isNormalised = true;
        end % performNormalization
        
        %
        % extract feature and class values from the table, return as [x, y]
        % datasetTable can be one of the class members datasetContentsAsTable,
        % trainTable or testTable
        % returns 2 tables: x training features, y expected values
        function [x, y] = extractXYFromTable(~, datasetTable)
          x = datasetTable(:,2:end);
          y = datasetTable(:,1);
        end
        
        %% Display Methods
        %

        %%
        % Display dataset summary information 
        function displayDatasetInformation(obj)
          normText = "(~normalised)";
          if obj.isNormalised
            normText = "(normalised)";
          end
          disp(obj);
          disp("Training Table Summary:");
          disp("=======================");
          summary(obj.trainTable);          
        end
        %%
        % Display plots of the dataset
        function displayDatasetPlots(obj)
          normText = "(~normalised)";
          if obj.isNormalised
            normText = "(normalised)";
          end
          %% Display sample's target values distribution to confirm it is equally distributed:
          obj.plotLetterDistribution(obj.trainTable, "Distribution of Classes " + normText);
          %% display correlation of attributes as a heatmap:
          obj.displayCorrelation(obj.trainTable, "Correlation " + normText)
          %% Display a grid comparing the attributes by plotting attributes against each other.
          obj.displayScatterMatrix(obj.trainTable, "Scatter Matrix of Attributes " + normText);      
          %% Display Predictor levels
          obj.plotPredictorLevels(obj.trainTable, 'Predictor Levels');
        end %function          
        
        %
        % Display correlation of training data as two heatmaps:
        % - A Pearson correlation of the attribute values is displayed
        % - A correlation of the null hypothesis p-values is displayed
        % makes use of heatmap found on Matlab's fileexchange:
        % Ameya Deoras (2020). Customizable Heat Maps (https://www.mathworks.com/matlabcentral/fileexchange/24253-customizable-heat-maps), MATLAB Central File Exchange. Retrieved November 2, 2020.
        %
        function displayCorrelation(obj, datasetTable, plotTitle)
          [x, ~] = obj.extractXYFromTable(datasetTable);
          figure("Name", plotTitle);
          mat = table2array(x);
          [covariance, pValue] = corrcoef(mat);       
          
          %% Plot correlation of attributes
          % Use Cell Values 
          [hImage, hText, hXText] = heatmap2(covariance, obj.featureNames,obj.featureNames, ...
            '%0.2f', 'TickAngle', 45, 'Colorbar', true, 'ShowAllTicks', true);
          title(plotTitle);
          
          %% Plot correlation of p-values
          % print standard deviation of the variables in x
          uiFigure = figure("Name", plotTitle + ' p-values');
            
          [hImage, hText, hXText] = heatmap2(pValue, obj.featureNames,obj.featureNames, ...
            '%0.2f', 'TickAngle', 45, 'Colorbar', false, 'ShowAllTicks', true, 'Colormap', 'copper', "TextColor", 'white');
          title(plotTitle + ' p-values');
          
        end
        
        %
        % Display a scatter plot matrix of of the attributes (processor
        % intensive)
        %
        function displayScatterMatrix(obj, datasetTable, plotTitle)
          [x, y] = obj.extractXYFromTable(datasetTable);
          fig = figure("Name", plotTitle);
          mat = table2array(x);
          targetMat = table2array(y);
          xnames = obj.featureNames;
          ynames = xnames;
          color = lines(26);
          [h,ax,bigax] = gplotmatrix(mat, mat, targetMat, color, [], [], [], 'variable', xnames, ynames);
          title(plotTitle);
          xsize = size(x);
          numFeatures = xsize(1,2);
          legend1 = legend(bigax,'show', 'Location','northeastoutside');
          % Align scatterplot labels, remove tick labels
          for xy = 1:numFeatures
            ax(numFeatures, xy).XLabel.Rotation = 45;
            ax(numFeatures, xy).XTickLabel = [''];
            ax(numFeatures, xy).XLabel.HorizontalAlignment = 'right';
            ax(numFeatures, 1).YLabel.Rotation = 0;
            ax(numFeatures, 1).YTickLabel = [''];
            ax(numFeatures, 1).YLabel.HorizontalAlignment = 'right';
          end
          fig_Position = fig.Position;
          % make it bigger to display figures:
          fig_Position(3) = fig_Position(3)*1.5;
          fig_Position(4) = fig_Position(4)*1.5;
          fig.Position = fig_Position;
        end
        
        %
        % display a bar chart showing the frequency distribution of the 
        % examples' classes
        %
        function plotLetterDistribution(obj, datasetTable, plotTitle)
          [x, y] = obj.extractXYFromTable(datasetTable);
          figure("Name", "Distribution of Classes");
          tabulated = tabulate(table2array(y));
          t = cell2table(tabulated,'VariableNames', ...
            {'Value','Count','Percent'});
          t.Value = categorical(t.Value);
          barh(t.Value,t.Count); %, 'BarWidth',3);
          ylabel('Letter');
          xlabel('Total Entries');
          title(plotTitle);
          fprintf("Summary of letter distribution:\n");
          s = summary(t);
          fprintf("Minimum examples per class: %d\nMaximum examples per class: %d\nMedian number of examples per class: %0.04f\n", ...
            s.Count.Min, s.Count.Max, s.Count.Median);
        end
                
        %
        % Plot levels
        %
        function plotPredictorLevels(obj, datasetTable, plotTitle)
          [X, Y] = obj.extractXYFromTable(datasetTable);
          countLevels = @(x)numel(categories(categorical(x)));
          numLevels = varfun(countLevels,X,'OutputFormat','uniform');
          
          figure('Name', plotTitle);
          bar(numLevels);
          title('Number of Levels Among Predictors');
          xlabel('Predictor variable');
          ylabel('Number of levels');
          h = gca;
          h.XTickLabel = obj.featureNames;
          h.XTickLabelRotation = 45;
          h.TickLabelInterpreter = 'none';
          title(plotTitle);
        end
        
    end % methods
    
    %
    % Static methods
    %
    methods(Static)
      
    end % static methods
    
end % class