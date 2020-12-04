classdef PlotUtil 
  properties
  end
  methods
  end
  methods(Static)

    %
    % plot the average training and test losses from the results table
    % ranged 0 .. 1, by result table row number
    function plotLossTestTrainComparison(resultsTable, plotTitle)
      PlotUtil.plotLossTestTrainComparisonWithYMax(resultsTable, plotTitle, 1.0);
    end  

    %
    % plot the average training and test losses from the results table
    % ranged 0 .. yMax, by result table row number
    function plotLossTestTrainComparisonWithYMax(resultsTable, plotTitle, yMax)
      % plot the 
      trainLossMeasure = resultsTable.avgTrainLoss;
      testLossMeasure = resultsTable.avgTestLoss;
      fig = figure("Name", plotTitle);
      ax = axes('Parent',fig);
      hold(ax,'on');
      %xlim(ax,[0 (size(obj.resultsTable, 1) + 1)]);
      ylim(ax,[0.0 yMax]);
      legend1 = legend(ax,'show');
      pd = plot(trainLossMeasure, 'Color',[0.4660 0.6740 0.1880], ...
        'DisplayName','Train','MarkerSize',15,'Marker','.');
      pd = plot(testLossMeasure, 'Color',[0.3010 0.7450 0.9330], ...
        'DisplayName','Test','MarkerSize',15,'Marker','.');
      xlabel("Result table row No.");
      ylabel("Loss measure");
      title(plotTitle);
      grid on;
    end 

    %
    % Plot the performance metrics: average accuracy, precision, recall, F1
    % by result table row number
    function plotMetrics(resultsTable, plotTitle)
      fig = figure("Name", plotTitle);
      ax = axes('Parent', fig);
      hold(ax, 'on');
      legend1 = legend(ax, 'show');
      acc = resultsTable.avgAccuracy;
      prec = resultsTable.avgPrecision;
      rec = resultsTable.avgRecall;
      f1 = resultsTable.avgF1;
      plot(acc, 'Color', [0.6760 0.6740 0.1880], 'DisplayName','Accuracy','MarkerSize',15,'Marker','.');
      plot(prec, 'Color', [0.4660 0.6740 0.6780], 'DisplayName','Precision','MarkerSize',15,'Marker','.');
      plot(rec, 'Color', [0.4660 0.1880 0.6740], 'DisplayName','Recall','MarkerSize',15,'Marker','.');
      plot(f1, 'Color', [0.4660 0.6740 0.1880], 'DisplayName','F1','MarkerSize',15,'Marker','.');
      xlabel("Result table row No.");
      ylabel("Metric Value");
      title(plotTitle); 
      grid on;
    end

    %
    % Plot average out of bag error by result table row number.
    function plotOob(resultsTable, plotTitle)
      fig = figure("Name", plotTitle);
      ax = axes('Parent', fig);
      hold(ax, 'on');
      %ylim(ax, [0.0 yMax]);
      legend1 = legend(ax, 'show');
      if ismember('avgOobLoss', resultsTable.Properties.VariableNames)
        oob = resultsTable.avgOobLoss;
        plot(oob, 'Color', [0.4660 0.6740 0.1880], 'DisplayName','Oob Error','MarkerSize',15,'Marker','.');    
      end  
      ylabel("Out of Bag Loss");
      xlabel("Result table row No.");
      title(plotTitle); 
      grid on;
    end %method
    
    function plotTime(resultsTable, plotTitle)
            fig = figure("Name", plotTitle);
      ax = axes('Parent', fig);
      hold(ax, 'on');
      legend1 = legend(ax, 'show');
      times = resultsTable.elapsedTime;
      predictTimes = resultsTable.predictTime;
      plot(times, 'Color', [0.6740 0.4660  0.1880], 'DisplayName','Train&Predict','MarkerSize',15,'Marker','.');    
      plot(predictTimes, 'Color', [0.4660 0.1880 0.6740 ], 'DisplayName',' Mean Predict','MarkerSize',15,'Marker','.');    
      ylabel("Elapsed time (seconds)");
      xlabel("Result table row No.");
      title(plotTitle); 
      grid on;
    end
  
  end % methods static
end %class