classdef LetterDecisionTreeResults < handle
  %LETTERDECISIONTREERESULTS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    resultsTable = {};
    resultsColumnNames = ["numberOfHoldOutRun" "trainValidateProportion" ...
      "maxNumSplit" "splitCriterion" "avgTrainAccuracy" "avgTestAccuracy" "misclassifiedEntryCount" "entryCount" "elapsedTime"];
        % temporary file to store the results:
    outputResultsFilename = "dtreeResultsFilenameNotSet.csv";
    fileHandle = -1;
  end
  
  methods
    
    %
    % Constructor
    %
    function obj = LetterDecisionTreeResults(csvResultsFilename)
      %create empty results table
      obj.resultsTable = table(obj.resultsColumnNames);
      obj.outputResultsFilename = csvResultsFilename;
    end
    
    %
    % Open temporary file to write results
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
           fprintf(obj.fileHandle, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", obj.resultsColumnNames(:));
    end
    
    function appendResult(obj,trainValidateProportion, maxNumSplit, splitCriterion, ...
                  numberOfHoldOutRun, avgTrainAccuracy, avgTestAccuracy, avgMisclassifiedEntryCount, entryCount, elapsedTime)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:appendResult", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fprintf(obj.fileHandle, "%d\t%0.02f\t%d\t%s\t%0.04f\t%0.04f\t%0.04f\t%d\t%0.04f\n", ...
               numberOfHoldOutRun, trainValidateProportion, ...
               maxNumSplit, splitCriterion, avgTrainAccuracy, avgTestAccuracy, avgMisclassifiedEntryCount, entryCount, elapsedTime);
    end
    
    function endGatheringResults(obj)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:endGatheringResults", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fclose(obj.fileHandle);
      obj.fileHandle = -1;
      obj.resultsTable = readtable(obj.outputResultsFilename, "Delimiter", "\t");
    end
    
    function plotCriteriaAccuracy(obj, plotTitle)
      tDeviance = obj.resultsTable(strcmp(obj.resultsTable.splitCriterion, 'deviance'), :);
      tTwoing = obj.resultsTable(strcmp(obj.resultsTable.splitCriterion, 'twoing'), :);
      tGdi = obj.resultsTable(strcmp(obj.resultsTable.splitCriterion, 'gdi'), :);
      % Create figure
      figure1 = figure("Name", plotTitle);
      % Create axes
      axes1 = axes('Parent',figure1);
      hold(axes1,'on');
      % Create plot
      pd = plot(tDeviance.avgTrainAccuracy,tDeviance.avgTestAccuracy,'DisplayName','deviance','MarkerSize',25,'Marker','.',...
        'LineStyle','none',...
        'Color',[0.9290 0.6940 0.1250]);
      pt = plot(tTwoing.avgTrainAccuracy,tTwoing.avgTestAccuracy,'DisplayName','twoing','MarkerSize',15,'Marker','.',...
        'LineStyle','none',...
        'Color',[0.4660 0.6740 0.1880]);
      pg = plot(tGdi.avgTrainAccuracy,tGdi.avgTestAccuracy,'DisplayName','gdi','MarkerSize',10,'Marker','.',...
        'LineStyle','none',...
        'Color',[0.3010 0.7450 0.9330]);
      
      % Uncomment the following line to preserve the X-limits of the axes
      xlim(axes1,[0.0 1.0]);
      ylim(axes1,[0.0 1.0]);
      grid on
      xlabel("Avg. Train Accuracy");
      ylabel("Avg. Test Accuracy");
      title(plotTitle);
      box(axes1,'on');
      hold(axes1,'off');
      % Create legend
      legend1 = legend(axes1,'show');
      set(legend1,...
      'Position',[0.139097200877803 0.799736733387832 0.190114065664802 0.0999999973009218]);
    end
    
    function plotAccuracyTestTrainComparison(obj)
      % plot the 
      trainAccuracyMeasure = obj.resultsTable.avgTrainAccuracy%abs(log((1 - obj.resultsTable.avgTrainAccuracy)).^2);
      testAccuracyMeasure = obj.resultsTable.avgTestAccuracy%abs(log((1 - obj.resultsTable.avgTestAccuracy)).^2);
      fig = figure("Name", "Accuracy by Result Row comparison");
      ax = axes('Parent',fig);
      hold(ax,'on');
      %xlim(ax,[0 (size(obj.resultsTable, 1) + 1)]);
      %ylim(ax,[0.0 (max(max([trainErrorMeasure, testErrorMeasure])) + 1)]);
      legend1 = legend(ax,'show');
      pd = plot(trainAccuracyMeasure, 'Color',[0.4660 0.6740 0.1880], 'DisplayName','Train','MarkerSize',15,'Marker','.');
      pd = plot(testAccuracyMeasure, 'Color',[0.3010 0.7450 0.9330], 'DisplayName','Test','MarkerSize',15,'Marker','.');
      xlabel("Result table row No.");
      ylabel("Accuracy measure");
      title("Accuracy by Result Row comparison");
    end
    
  end % methods
end % class

