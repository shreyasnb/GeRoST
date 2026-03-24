function ieee_settings()
% IEEE_SETTINGS Configures MATLAB's default plotting properties to adhere 
% to standard IEEE conference paper guidelines.
%
% This applies LaTeX interpreters, standard font sizes, and appropriate
% line widths to all subsequent figures generated in the session.

    % Set default text interpreters to LaTeX
    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');

    % Set default font sizes (10pt is standard for IEEE figures)
    set(groot, 'defaultAxesFontSize', 10);
    set(groot, 'defaultTextFontSize', 10);

    % Set default line widths for better visibility in print
    set(groot, 'defaultLineLineWidth', 1.5);
    set(groot, 'defaultStairLineWidth', 1.5);
    
    % Ensure axes borders are clean
    set(groot, 'defaultAxesLineWidth', 1.0);
    set(groot, 'defaultAxesBox', 'on');

    % Set figure background color to white
    set(groot, 'defaultFigureColor', 'w');
end