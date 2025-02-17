clear;
clc;

%% data input
load('Preprocessed_PHI014_FSit.mat');

time = data.time;
right = data.cbfv_r;
left = data.cbfv_l;

pressure = data.BP;

tsi = data.TSI;
tsiff = data.TSI_FF;

time10 = data.time_10Hz;
absO2Hb = data.absO2Hb;
absHHb = data.absHHb;


%% plot

figure;
plot(time, right, 'r', 'LineWidth', 1.5); hold on;
plot(time, left, 'b', 'LineWidth', 1.5);
xlabel('Time');
ylabel('CBFV');
legend({'Right', 'Left'});
title('CBFV Right vs. Left');
grid on;

% boxplot
figure;
boxchart([ones(size(right)); 2*ones(size(left))], [right; left])
xticks([1 2])
xticklabels({'Right', 'Left'})
ylabel('CBFV')
title('Comparison of CBFV Right vs. Left')
grid on;


diff_val = right - left;

figure;
plot(time, diff_val, 'k', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Difference (Right - Left)');
title('Difference between Right and Left CBFV');
grid on;



figure;
histogram(right, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5); hold on;
histogram(left, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5);
xlabel('CBFV');
ylabel('Frequency');
legend({'Right', 'Left'});
title('Histogram of CBFV Right vs. Left');
grid on;


figure;
qqplot(right);
title('Q-Q Plot for right');

figure;
plot(time, right, 'r', 'LineWidth', 1.5); hold on;
xlabel('Time');
ylabel('Blood Pressure');
legend({'Blood'});
title('Blood Pressure Changes with Time');
grid on;


time2 = time(1:2:end);
figure;
plot(time2, tsi(:,1), 'r', 'LineWidth', 1.5); hold on;
plot(time2, tsi(:,2), 'b', 'LineWidth', 1.5);

xlabel('Time(s)');
ylabel('TSI(%)');
legend({'TSI\_left', 'TSI\_right'});
title('TSI Changes with Time');
grid on;


figure;
plot(time2, tsiff(:,1), 'r', 'LineWidth', 1.5); hold on;
plot(time2, tsiff(:,2), 'b', 'LineWidth', 1.5);
xlabel('Time(s)');
ylabel('tsiff(%)');
legend({'tsiff\_left', 'tsiff\_right'});
title('Tsiff Changes with Time');
grid on;

% figure;
% plot(time2, absO2Hb, 'r', 'LineWidth', 1.5); hold on;
% plot(time2, absHHb, 'b', 'LineWidth', 1.5);
% plot(time2, absHHb+absO2Hb, 'k', 'LineWidth', 1.5);
% ttsi = absO2Hb./(absHHb+absO2Hb)*100;
% plot(time2, ttsi, 'green', 'LineWidth', 1.5)
% plot(time2, tsi(:,2), 'red', 'LineWidth', 1.0);
% xlabel('Time');
% ylabel('umol');
% legend({'O2Hb','HHb','Total_Hb','TSI'});
% title('Tsiff Changes with Time');
% grid on;


figure;
plot(time2, absO2Hb, 'r', 'LineWidth', 1.5); hold on;
plot(time2, absHHb, 'b', 'LineWidth', 1.5);
xlabel('Time(s)');
ylabel('Concentration (umol/L)');
legend({'O2Hb','HHb'});
title('O2Hb and HHb Changes with Time');
grid on;

figure;
plot(time2, absO2Hb, 'r', 'LineWidth', 1.5); hold on;
xlabel('Time(s)');
ylabel('Concentration (umol/L)');
legend({'O2Hb'});
title('O2Hb Changes with Time');
grid on;

figure;
plot(time2, absHHb, 'b', 'LineWidth', 1.5);hold on;
xlabel('Time(s)');
ylabel('Concentration (umol/L)');
legend({'HHb'});
title('HHb Changes with Time');
grid on;
