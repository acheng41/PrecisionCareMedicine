%% Test
% test different functions and visualize data
% By: Samuel Bello
% Created: 6/13/24
% Last Updated: 6/14/24

clear
close all

%% Load Joint Angles and Positions
load data\080624\gait_recording_080624_walk.mat
color_r = 'red';
color_l = 'blue';

start_idx = 316;
stop_idx = 544;
start_t = t_insole_l(start_idx);
stop_t = t_insole_l(stop_idx);

%% Animate Gait
maxLineLen = 100; % Number of samples displayed by the animated lines
fig = figure('units','normalized','outerposition',[0 0 1 1]);
p1 = subplot(3,9,[1,2,10,11,19,20]);
p2 = subplot(3,9,[3,4,12,13,21,22]);
p3 = subplot(3,9,[5,6,14,15,23,24]);

p4 = subplot(3,9,7);
hold on
an1 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an2 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p4,[-90 90])
xlabel(p4,'Time (s)');
ylabel(p4,'Angle (deg)');
title(p4,'Hip Adduction/Abduction');

p5 = subplot(3,9,8);
hold on
an3 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an4 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p5,[-90 90])
xlabel(p5,'Time (s)');
ylabel(p5,'Angle (deg)');
title(p5,'Hip Internal/External');

p6 = subplot(3,9,9);
hold on
an5 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an6 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p6,[-90 90])
xlabel(p6,'Time (s)');
ylabel(p6,'Angle (deg)');
title(p6,'Hip Flexion/Extension');


p7 = subplot(3,9,16);
hold on
an7 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an8 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p7,[-90 90])
xlabel(p7,'Time (s)');
ylabel(p7,'Angle (deg)');
title(p7,'Knee Adduction/Abduction');

p8 = subplot(3,9,17);
hold on
an9 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an10 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p8,[-90 90])
xlabel(p8,'Time (s)');
ylabel(p8,'Angle (deg)');
title(p8,'Knee Internal/External');

p9 = subplot(3,9,18);
hold on
an11 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an12 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p9,[-90 90])
xlabel(p9,'Time (s)');
ylabel(p9,'Angle (deg)');
title(p9,'Knee Flexion/Extension');


p10 = subplot(3,9,25);
hold on
an13 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an14 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p10,[-90 90])
xlabel(p10,'Time (s)');
ylabel(p10,'Angle (deg)');
title(p10,'Ankle Inversion/Eversion');

p11 = subplot(3,9,26);
hold on
an15 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an16 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p11,[-90 90])
xlabel(p11,'Time (s)');
ylabel(p11,'Angle (deg)');
title(p11,'Ankle Adduction/Abduction');

p12 = subplot(3,9,27);
hold on
an17 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_r);
an18 = animatedline('MaximumNumPoints',maxLineLen,'Color',color_l);
hold off
ylim(p12,[-90 90])
xlabel(p12,'Time (s)');
ylabel(p12,'Angle (deg)');
title(p12,'Ankle Dorsi/Plantar Flexion');

% initialize gait video frame
jnt_pos = cell2mat(jnt_pos_all_l(1));
lim_x = [min(jnt_pos(:,1)) max(jnt_pos(:,1))];
lim_y = [min(jnt_pos(:,3)) max(jnt_pos(:,3))];
lim_z = [min(jnt_pos(:,2)) max(jnt_pos(:,2))];
for i = 2:length(jnt_pos_all_l)
    jnt_pos = cell2mat(jnt_pos_all_l(i));
    lim_x = [min(lim_x(1), min(jnt_pos(:,1))) max(lim_x(2), max(jnt_pos(:,1)))];
    lim_y = [min(lim_y(1), min(jnt_pos(:,3))) max(lim_y(2), max(jnt_pos(:,3)))];
    lim_z = [min(lim_z(1), min(jnt_pos(:,2))) max(lim_z(2), max(jnt_pos(:,2)))];
end
for i = 1:length(jnt_pos_all_r)
    jnt_pos = cell2mat(jnt_pos_all_r(i));
    lim_x = [min(lim_x(1), min(jnt_pos(:,1))) max(lim_x(2), max(jnt_pos(:,1)))];
    lim_y = [min(lim_y(1), min(jnt_pos(:,3))) max(lim_y(2), max(jnt_pos(:,3)))];
    lim_z = [min(lim_z(1), min(jnt_pos(:,2))) max(lim_z(2), max(jnt_pos(:,2)))];
end
% lim_x = [lim_x(1) - .5, lim_x(2) + .5];
% lim_y = [lim_y(1) - .5, lim_y(2) + .5];
% lim_z = [lim_z(1) - .5, lim_z(2) + .5];`

lim_x = [-1.11 -0.36];
lim_z = [0.06 1.31];

% pause(.5)
for i = 1:length(t_insole_l)%[325 358 368 374 401 ]%start_idx:stop_idx
    [~,idx] = min(abs(t_trackers - t_insole_l(i)));

    pos_l = cell2mat(jnt_pos_all_l(idx));
    plot_jnt_pos_l = [pos_l(1,1), pos_l(1,3), pos_l(1,2);
        pos_l(2,1), pos_l(2,3), pos_l(2,2)
        pos_l(3,1), pos_l(3,3), pos_l(3,2)
        pos_l(4,1), pos_l(4,3), pos_l(4,2)];
    jnt_angles_l = cell2mat(jnt_angles_all_l(idx+1));

    pos_r = cell2mat(jnt_pos_all_r(idx));
    plot_jnt_pos_r = [pos_r(1,1), pos_r(1,3), pos_r(1,2);
        pos_r(2,1), pos_r(2,3), pos_r(2,2)
        pos_r(3,1), pos_r(3,3), pos_r(3,2)
        pos_r(4,1), pos_r(4,3), pos_r(4,2)];
    jnt_angles_r = cell2mat(jnt_angles_all_r(idx+1));

    try
        insole_l = insoleAll_l(i,:)';
        img = reshape(insole_l,[64 16]);
        axes(p2)
        imagesc(img)
        title("Left Insole")
    end

    try
        insole_r = insoleAll_r(i,:)';
        img = reshape(insole_r,[64 16]);
        img = fliplr(img);
        img(1:32,:) = flipud(img(1:32,:));
        axes(p3)
        imagesc(img)
        title("Right Insole")
    end

    try
        plotStickFigure3(p1,plot_jnt_pos_r,plot_jnt_pos_l,color_r,color_l,lim_x,lim_y,lim_z)
        addpoints(an1,t_trackers(idx),jnt_angles_r(1,1));
        addpoints(an2,t_trackers(idx),jnt_angles_l(1,1));
        
        addpoints(an3,t_trackers(idx),jnt_angles_r(1,2));
        addpoints(an4,t_trackers(idx),jnt_angles_l(1,2));

        addpoints(an5,t_trackers(idx),jnt_angles_r(1,3));
        addpoints(an6,t_trackers(idx),jnt_angles_l(1,3));

        addpoints(an7,t_trackers(idx),jnt_angles_r(2,1));
        addpoints(an8,t_trackers(idx),jnt_angles_l(2,1));

        addpoints(an9,t_trackers(idx),jnt_angles_r(2,2));
        addpoints(an10,t_trackers(idx),jnt_angles_l(2,2));

        addpoints(an11,t_trackers(idx),jnt_angles_r(2,3));
        addpoints(an12,t_trackers(idx),jnt_angles_l(2,3));

        addpoints(an13,t_trackers(idx),jnt_angles_r(3,1));
        addpoints(an14,t_trackers(idx),jnt_angles_l(3,1));

        addpoints(an15,t_trackers(idx),jnt_angles_r(3,2));
        addpoints(an16,t_trackers(idx),jnt_angles_l(3,2));

        addpoints(an17,t_trackers(idx),jnt_angles_r(3,3));
        addpoints(an18,t_trackers(idx),jnt_angles_l(3,3));
        drawnow limitrate;
    end
    % pause(.01)
end