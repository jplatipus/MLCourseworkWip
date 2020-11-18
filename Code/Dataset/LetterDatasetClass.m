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
      % flag indicating if the dataset has been normalised
      isNormalised = false;
      % flag indicating if the dataset has been standardised: 
      % (Center and scale to have mean 0 and standard deviation 1)
      isStandardised = false;
      
    end % properties
    
    methods
        
        % Public
        % Constructor: load dataset, split data into class members (tables)
        %
        function obj = LetterDatasetClass(normalised, standardized)
          %Load file: 
          obj.datasetContentsAsTable = readtable(obj.datasetFilePath, 'Delimiter', ',');
          % normalise dataset
          if normalised
            obj.datasetContentsAsTable = normalize(obj.datasetContentsAsTable, ...
              'norm', Inf, 'DataVariables', @isnumeric);
            obj.isNormalised = true;
          else
            if standardized
              obj.datasetContentsAsTable = normalize(obj.datasetContentsAsTable, ...
                'zscore', 'std', 'DataVariables', @isnumeric);
              obj.isStandardised = true;
            end
          end
          
          %Extract set of class values
          obj.validClassValues = unique(obj.datasetContentsAsTable(:,1));
          %Split dataset into train and test sets
          rng(obj.randomSeed)
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
        end
        
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
        % Display dataset information
        function displayDatasetInformation(obj)
          normText = "(~normalised)";
          if obj.isNormalised
            normText = "(normalised)";
          else 
            if obj.isStandardised
              normText = "(standardised)";
            end
          end
          disp(obj);
          disp("Training Table Summary:");
          disp("=======================");
          summary(obj.trainTable);
          %% Display sample's target values distribution to confirm it is equally distributed:
          obj.plotLetterDistribution(obj.trainTable, "Distribution of Classes " + normText);
          %% display correlation of attributes as a heatmap:
          obj.displayCorrelation(obj.trainTable, "Correlation " + normText)
          %% Display a grid comparing the attributes by plotting attributes against each other.
          obj.displayScatterMatrix(obj.trainTable, "Scatter Matrix of Attributes " + normText);
          %% Display Dataset PCA
          obj.plotPCA(obj.trainTable, "Principle Component Analysis " + normText);
          %% Display parallel coordinates plot of each class, value and feature
          obj.plotParallelCoordinates(obj.trainTable, "Parallel Coordinates Plot "  + normText);
        end %function          
        
        %
        % Display correlation of training data as two heatmaps:
        % - A correlation of the attribute values is displayed
        % - A correlation of the null hypothesis p-values is displayed
        % make use of heatmap found on Matlab's fileexchange:
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
          figure("Name", plotTitle);
          mat = table2array(x);
          targetMat = table2array(y);
          xnames = obj.featureNames;
          ynames = xnames;
          color = lines(26);
          [h,ax,bigax] = gplotmatrix(mat, mat, targetMat, color, [], [], [], 'variable', xnames, ynames);
          title(plotTitle);
          % Align scatterplot labels, remove tick labels
          for xy = 1:16
            ax(16, xy).XLabel.Rotation = 45;
            ax(16, xy).XTickLabel = [''];
            ax(16, xy).XLabel.HorizontalAlignment = 'right';
            ax(xy, 1).YLabel.Rotation = 0;
            ax(xy, 1).YTickLabel = [''];
            ax(xy, 1).YLabel.HorizontalAlignment = 'right';
          end
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
          summary(t);
        end
         
        %
        % Plot PCA of training data features
        %
        function plotPCA(obj, datasetTable, plotTitle)
            [x, y] = obj.extractXYFromTable(datasetTable);
            figure("Name", "PCA Chart");
            [coeff,score,latent,tsquared,explained] = pca(table2array(x));
            disp(explained);
            bar(explained(1:16,:));
            xticks(1:16);
            xlabel("Components in order of importance");
            ylabel("Percentage variability");
            xtickangle(45);
            grid on;
            title(plotTitle);
        end;  
        
        %
        % Plot a parallel coordinates plot of the measuements for each
        % feature in the given dataset (training is used).
        % The data is normalised then plotted, using a different colour for
        % each class (letter).
        % In order to make the plot less busy, only the median (solid
        % line), 25% quartile (dotted line below its respective solid line) and 
        % 75% quartile (dotted line above its respective solid line) 
        % are plotted for each class.
        %
        function plotParallelCoordinates(obj, datasetTable, plotTitle)
          [x, y] = obj.extractXYFromTable(datasetTable);
          x = table2array(x);
          y = table2array(y);
          meanX = mean(x);
          stdX = std(x);
          normalisedX = (x - meanX) ./ stdX;
          figure('Name', plotTitle);
          labels = datasetTable.Properties.VariableNames(:,2:end);
          parallelcoords(normalisedX, 'group', y, 'labels', labels, 'quantile', 0.25);
          xtickangle(45);
          title(plotTitle);
        end;
        
    end % methods
    
    %
    % Static methods
    %
    methods(Static)
      
    end % static methods
    
end % class