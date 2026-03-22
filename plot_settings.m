%% IEEE Conference Paper Plot Settings
% Standard font sizes for IEEE publications:
% - Figure width: typically 3.5 inches (single column) or 7.25 inches (double column)
% - Axes font: 10pt
% - Axis labels: 11pt (bold)
% - Titles: 12pt (bold)
% - Legend: 10pt

function settings = plot_settings()
    % Returns a structure with IEEE-standard plot settings
    
    settings.fig_width = 7;      % inches, for double-column layout
    settings.fig_height = 5.25;  % inches
    settings.font_main = 10;     % main axis text
    settings.font_label = 11;    % axis labels
    settings.font_title = 12;    % subplot titles
    settings.font_legend = 10;   % legend
    settings.line_width = 1.5;   % line width for plots
    settings.marker_size = 6;    % marker size
    settings.interpreter = 'latex'; % text interpreter
    
    % Figure size in pixels (for display)
    settings.fig_pos = [100, 100, 1400, 900];
end
