function [gait] = get_gait_parameters_insole2_v2(insole_r,insole_l,t_r,t_l)
gait = struct();
gait.t_r = t_r;
gait.insole_r = insole_r;
gait.t_l = t_l;
gait.insole_l = insole_l;
gait.area = 0.002^2;
gait.dim = [sqrt(size(insole_r,2)) * 2, sqrt(size(insole_r,2)) / 2];


%% Instantaneous Peak Pressure, Pressure-Time Intergral
gait.pp_r = max(insole_r,[],2);
gait.pp_l = max(insole_l,[],2);
gait.pp_x_r = zeros(size(t_r));
gait.pp_y_r = zeros(size(t_r));
gait.pp_x_l = zeros(size(t_l));
gait.pp_y_l = zeros(size(t_l));

gait.pti_r = cumtrapz(insole_r,2);
gait.pti_l = cumtrapz(insole_r,2);
gait.fti_r = cumtrapz(sum(insole_r .* gait.area,2));
gait.fti_l = cumtrapz(sum(insole_l .* gait.area,2));

%% Center of Pressure, Gait Trajectory, Contact Area and Trace
gait.foot_trace_r = zeros(length(t_r),1);
gait.foot_trace_l = zeros(length(t_l),1);
gait.cop_x_r = zeros(length(t_r),1);
gait.cop_y_r = zeros(length(t_r),1);
gait.cop_x_l = zeros(length(t_l),1);
gait.cop_y_l = zeros(length(t_l),1);
gait.cont_area_r = zeros(length(t_r),1);
gait.cont_area_l = zeros(length(t_l),1);


for i = 1:length(t_r)
    frame = insole_r(i,:);
    frame = reshape(frame,gait.dim(1),gait.dim(2));
    frame = fliplr(frame);
    frame(1:gait.dim(1) / 2,:) = flipud(frame(1:gait.dim(1) / 2,:));
    gait.foot_trace_r(i) = mean(frame,'all'); % trace is the average pressure for each frame
    [x,y] = find(frame);
    idx = sub2ind(size(frame),x,y);
    gait.cop_x_r(i) = sum(x .* frame(idx)) / sum(frame(idx)); % cop x-coordinate
    gait.cop_y_r(i) = sum(y .* frame(idx)) / sum(frame(idx)); % cop y-coordinate
    gait.cont_area_r(i) = length(x); % contact gait.area
    
    % pp x and y coordinate
    [x,y] = find(frame == gait.pp_r(i));
    if length(x) > 1
        gait.pp_x_r(i) = mean(x);
        gait.pp_y_r(i) = mean(y);
    else
        gait.pp_x_r(i) = x;
        gait.pp_y_r(i) = y;
    end
end

for i = 1:length(t_l)
    frame = insole_l(i,:);
    frame = reshape(frame,gait.dim(1),gait.dim(2));
    frame(1:gait.dim(1) / 2,:) = flipud(frame(1:gait.dim(1) / 2,:));
    gait.foot_trace_l(i) = mean(frame,'all');
    [x,y] = find(frame);
    idx = sub2ind(size(frame),x,y);
    gait.cop_x_l(i) = sum(x .* frame(idx)) / sum(frame(idx));
    gait.cop_y_l(i) = sum(y .* frame(idx)) / sum(frame(idx));
    gait.cont_area_l(i) = length(x);

    [x,y] = find(frame == gait.pp_l(i));
    if length(x) > 1
        gait.pp_x_l(i) = round(mean(x));
        gait.pp_y_l(i) = round(mean(y));
    else
        gait.pp_x_l(i) = x;
        gait.pp_y_l(i) = y;
    end
end

% plot COP over time
% figure()
% for i = 1:max(length(t_r),length(t_l))
%     try
%         frame = insole_l(i,:);
%         frame = reshape(frame,gait.dim(1),gait.dim(2));
%         frame(1:gait.dim(1) / 2,:) = flipud(frame(1:gait.dim(1) / 2,:));
%         subplot(1,2,1)
%         imagesc(frame)
%         hold on
%         plot(gait.cop_y_l(i),gait.cop_x_l(i), '.', 'MarkerSize',30,'Color','red')
%         plot(gait.pp_y_l(i),gait.pp_x_l(i), '*', 'MarkerSize',10,'Color','black')
%         hold off
%         title("Left Insole COP")
%         legend("Center of Pressure (COP)","Peak Pressure (PP)")
%         drawnow limitrate
%     end
%     try
%         frame = insole_r(i,:);
%         frame = reshape(frame,gait.dim(1),gait.dim(2));
%         frame = fliplr(frame);
%         frame(1:gait.dim(1) / 2,:) = flipud(frame(1:gait.dim(1) / 2,:));
%         subplot(1,2,2)
%         imagesc(frame)
%         hold on
%         plot(gait.cop_y_r(i),gait.cop_x_r(i), '.', 'MarkerSize',30,'Color','red')
%         plot(gait.pp_y_r(i),gait.pp_x_r(i), '*', 'MarkerSize',10,'Color','black')
%         hold off
%         title("Right Insole COP")
%         legend("Center of Pressure (COP)","Peak Pressure (PP)")
%         drawnow limitrate
%     end
%     pause(0.1)
% end

% plot contact area
figure()
set(gcf,'color','white');
subplot(1,2,1)
plot(t_l,gait.cont_area_l)
title("Left Insole Contact area")
xlabel("Time (s)")
ylabel("Area (pixels)")

subplot(1,2,2)
plot(t_r,gait.cont_area_r)
title("Right Insole Contact area")
xlabel("Time (s)")
ylabel("Area (pixels)")

%% Heel Strike Toe Off and Related Parameters
% get initial contact and foot off
thresh_r = min(gait.foot_trace_r) + 0.1 * range(gait.foot_trace_r);
thresh_l = min(gait.foot_trace_l) + 0.1 * range(gait.foot_trace_l);
gait.strike_r = [];
gait.off_r = [];
gait.strike_l = [];
gait.off_l = [];
% right
for i = 2:length(t_r) - 1
    if gait.foot_trace_r(i) >= thresh_r && gait.foot_trace_r(i-1) < thresh_r
        gait.strike_r = [gait.strike_r; i - 1];
    end
    if gait.foot_trace_r(i) >= thresh_r && gait.foot_trace_r(i+1) < thresh_r
        gait.off_r = [gait.off_r; i + 1];
    end
end
% left
for i = 2:length(t_l) - 1
    if gait.foot_trace_l(i) >= thresh_l && gait.foot_trace_l(i-1) < thresh_l
        gait.strike_l = [gait.strike_l; i - 1];
    end
    if gait.foot_trace_l(i) >= thresh_l && gait.foot_trace_l(i+1) < thresh_l
        gait.off_l = [gait.off_l; i + 1];
    end
end

% isolate only complete gait cycles (remove  toe offs before first heel strike or after last strike)
gait.off_r = gait.off_r(gait.off_r > gait.strike_r(1));
gait.off_r = gait.off_r(gait.off_r <= gait.strike_r(end));
gait.off_l = gait.off_l(gait.off_l > gait.strike_l(1));
gait.off_l = gait.off_l(gait.off_l <= gait.strike_l(end));

% cycle duration
gait.cycle_dur_r = diff(t_r(gait.strike_r));
gait.cycle_dur_l = diff(t_l(gait.strike_l));

% cycle duration variability
gait.cycle_var_r = std(gait.cycle_dur_r) / mean(gait.cycle_dur_r) * 100;
gait.cycle_var_l = std(gait.cycle_dur_l) / mean(gait.cycle_dur_l) * 100;

% cadence
gait.cadence = min(length(gait.cycle_dur_r),length(gait.cycle_dur_l)) / min(sum(gait.cycle_dur_r),sum(gait.cycle_dur_l)) * 60;

% stance phase
gait.stance_r = (t_r(gait.off_r) - t_r(gait.strike_r(1:end - 1))) ./ gait.cycle_dur_r * 100;
gait.stance_l = (t_l(gait.off_l) - t_l(gait.strike_l(1:end - 1))) ./ gait.cycle_dur_l * 100;

% swing phase
gait.swing_r = 100 - gait.stance_r;
gait.swing_l = 100 - gait.stance_l;

% Asymmetry (swing)
gait.asym = (mean(gait.swing_l) - mean(gait.swing_r)) / (0.5 * (mean(gait.swing_l) + mean(gait.swing_r))) * 100;

figure()
set(gcf,'color','white');
subplot(1,2,1)
plot(t_l, gait.foot_trace_l)
hold on
plot(t_l(gait.strike_l),gait.foot_trace_l(gait.strike_l),'o','Color','red')
plot(t_l(gait.off_l),gait.foot_trace_l(gait.off_l),'o','Color','green')
yline(thresh_l,'Color','black')
hold off
title("Time Trace of Left Foot")
xlabel("Frame")
ylabel("Intensity")
legend("Foot Trace","Initial Contact","Foot off","Threshold")

subplot(1,2,2)
plot(t_r, gait.foot_trace_r)
hold on
plot(t_r(gait.strike_r),gait.foot_trace_r(gait.strike_r),'o','Color','red')
plot(t_r(gait.off_r),gait.foot_trace_r(gait.off_r),'o','Color','green')
yline(thresh_r,'Color','black')
hold off
title("Time Trace of Right Foot")
xlabel("Time (s)")
ylabel("Intensity")
legend("Foot Trace","Initial Contact","Foot off","Threshold")

% plot gait trajectory during stance
range_img_r = range(insole_r,1);
range_img_r = reshape(range_img_r,gait.dim(1),gait.dim(2));
range_img_r = fliplr(range_img_r);
range_img_r(1:gait.dim(1) / 2,:) = flipud(range_img_r(1:gait.dim(1) / 2,:));

range_img_l = range(insole_l,1);
range_img_l = reshape(range_img_l,gait.dim(1),gait.dim(2));
range_img_l(1:gait.dim(1) / 2,:) = flipud(range_img_l(1:gait.dim(1) / 2,:));

figure()
set(gcf,'color','white');
subplot(1,2,1)
imagesc(range_img_l)
hold on
for i = 1:length(gait.strike_l) - 1
    gait_traj_x_l = gait.cop_x_l(gait.strike_l(i):gait.off_l(i));
    gait_traj_y_l = gait.cop_y_l(gait.strike_l(i):gait.off_l(i));
    plot(gait_traj_y_l,gait_traj_x_l,'LineWidth',2,'Color','red')
end
hold off
title("Range Summary Image and Gait Trajectory of Left Insole")

subplot(1,2,2)
imagesc(range_img_r)
hold on
for i = 1:length(gait.strike_r) - 1
    gait_traj_x_r = gait.cop_x_r(gait.strike_r(i):gait.off_r(i));
    gait_traj_y_r = gait.cop_y_r(gait.strike_r(i):gait.off_r(i));
    plot(gait_traj_y_r,gait_traj_x_r,'LineWidth',2,'Color','red')
end
hold off
title("Range Summary Image and Gait Trajectory of Right Insole")
end



