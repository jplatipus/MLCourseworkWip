classdef LetterDecisionTreeResults < handle
  % Class used to write results to a csv file, load the results csv into an
  % instance of this class, and functionality to append a row to the
  % results
  
  properties
    resultsTable = {};
    resultsColumnNames = ["numberOfFolds", "minLeafSize", "minParentSize",...
      "maxNumSplit", "splitCriterion", ...
      "avgTrainLoss", "avgTestLoss", ...
      "avgAccuracy", "avgPrecision", "avgRecall", "avgF1", ...
      "entryCount", "elapsedTime", "predictTime", "randomSeed"];
        % temporary file to store the results:
    outputResultsFilename = "dtreeResultsFilenameNotSet.csv";
    fileHandle = -1;
  end
  
  methods
    
    %%
    % Constructor
    %
    function obj = LetterDecisionTreeResults(csvResultsFilename)
      %create empty results table
      obj.resultsTable = table(obj.resultsColumnNames);
      obj.outputResultsFilename = csvResultsFilename;
    end
    
    %%
    % Open csv file to write results, output header
    % throws exception if error
    function startGatheringResults(obj)
       [h, msg] = fopen(obj.outputResultsFilename, 'w');
       if h == -1
           fprintf("Error opening output file %s\n", msg);
           exception = MException("LetterDecisionTreeResults:startGatheringResults", "Error opening output file: %s\n", msg);
           throw(exception);
       end
       % output header
       obj.fileHandle = h;
       fprintf(obj.fileHandle, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", obj.resultsColumnNames(:));
    end
    
    %%
    % append a result row to the csv file
    %
    function appendResult(obj, minLeafSize, minParentSize, maxNumSplit, splitCriterion, ...
                  numberOfFolds, avgTrainLoss, avgTestLoss, ...
                  avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                  entryCount, elapsedTime, predictTime, randomSeed)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:appendResult", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fprintf(obj.fileHandle, "%d\t%d\t%d\t"+...
                              "%d\t%s\t%0.04f\t%0.04f\t"+...
                              "%0.04f\t%0.04f\t%0.04f\t%0.04f\t"+...
                              "%d\t%0.04f\t%0.04f\t%d\n", ...
               numberOfFolds, minLeafSize, minParentSize, ...
               maxNumSplit, splitCriterion, avgTrainLoss, avgTestLoss,  ...
              avgAccuracy, avgPrecision, avgRecall, avgF1, ...
              entryCount, elapsedTime, predictTime, randomSeed);
    end
    
    %%
    % close the csv file
    %
    function endGatheringResults(obj)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:endGatheringResults", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fclose(obj.fileHandle);
      obj.fileHandle = -1;
      obj.resultsTable = readtable(obj.outputResultsFilename, "Delimiter", "\t");
    end
    
    %%
    % Plot a comparison of the performance for the different split
    % criteria found in the results table
    %
    function plotCriteriaLoss(obj, plotTitle)
      tDeviance = obj.resultsTable(strcmp(obj.resultsTable.splitCriterion, 'deviance'), :);
      tTwoing = obj.resultsTable(strcmp(obj.resultsTable.splitCriterion, 'twoing'), :);
      tGdi = obj.resultsTable(strcmp(obj.resultsTable.splitCriterion, 'gdi'), :);
      % Create figure
      figure1 = figure("Name", plotTitle);
      % Create axes
      axes1 = axes('Parent',figure1);
      hold(axes1,'on');
      % Create plot
      pd = plot(tDeviance.avgTrainLoss,tDeviance.avgTestLoss,'DisplayName','deviance','MarkerSize',25,'Marker','.',...
        'LineStyle','none',...
        'Color',[0.9290 0.6940 0.1250]);
      pt = plot(tTwoing.avgTrainLoss,tTwoing.avgTestLoss,'DisplayName','twoing','MarkerSize',15,'Marker','.',...
        'LineStyle','none',...
        'Color',[0.4660 0.6740 0.1880]);
      pg = plot(tGdi.avgTrainLoss,tGdi.avgTestLoss,'DisplayName','gdi','MarkerSize',10,'Marker','.',...
        'LineStyle','none',...
        'Color',[0.3010 0.7450 0.9330]);
      
      % Uncomment the following line to preserve the X-limits of the axes
      xlim(axes1,[0.0 0.2]);
      ylim(axes1,[0.0 0.2]);
      grid on
      xlabel("Avg. Train Loss");
      ylabel("Avg. Test Loss");
      title(plotTitle);
      box(axes1,'on');
      hold(axes1,'off');
      % Create legend
      legend1 = legend(axes1,'show', 'Location','northeastoutside');
      %set(legend1,...
      %'Position',[0.139097200877803 0.799736733387832 0.190114065664802 0.0999999973009218]);
      % make it bigger to display figures:
      fig_Position = figure1.Position;
      fig_Position(3) = fig_Position(3)*1.5;
      fig_Position(4) = fig_Position(4)*1.5;
      figure1.Position = fig_Position;    
    end
  end % methods
  
  methods(Static)
    %%
    % creates an instance of this class by loading a csv file
    %
    function instance = getInstanceFromCsvResults(csvFilename)
      instance = LetterDecisionTreeResults(csvFilename);
      instance.resultsTable = readtable(csvFilename, "Delimiter", "\t");  
    end  % function  
    
  end % methods(Static)
end % class

