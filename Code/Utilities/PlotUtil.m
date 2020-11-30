classdef PlotUtil 
  properties
  end
  methods
  end
  methods(Static)

  function plotLossTestTrainComparison(resultsTable, plotTitle)
        % plot the 
        trainLossMeasure = resultsTable.avgTrainLoss;
        testLossMeasure = resultsTable.avgTestLoss;
        fig = figure("Name", plotTitle);
        ax = axes('Parent',fig);
        hold(ax,'on');
        %xlim(ax,[0 (size(obj.resultsTable, 1) + 1)]);
        ylim(ax,[0.0 1.0]);
        legend1 = legend(ax,'show');
        pd = plot(trainLossMeasure, 'Color',[0.4660 0.6740 0.1880], 'DisplayName','Train','MarkerSize',15,'Marker','.');
        pd = plot(testLossMeasure, 'Color',[0.3010 0.7450 0.9330], 'DisplayName','Test','MarkerSize',15,'Marker','.');
        xlabel("Result table row No.");
        ylabel("Loss measure");
        title(plotTitle);
        grid on;
  end  
    
  end % methods static
end %class