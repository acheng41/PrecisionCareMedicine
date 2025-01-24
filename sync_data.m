%% Sync Motion Trackers and Insole
% Shift time array of insoles so that it is synced with motion capture data
% By: Samuel Bello
% Created: 11/13/24
% Last Updated: 11/20/24

clear
close all

filename = "111724\gait_recording_111724_walk3.mat";
load(filename)


%% get time trace of insole data and tracker position
gait_insole = get_gait_parameters_insole2_v2(insoleAll_r,insoleAll_l,t_insole_r,t_insole_l);
gait_trackers = get_gait_parameters_vive2(jnt_pos_all_r,jnt_pos_all_l,jnt_angles_all_r,jnt_angles_all_l,t_trackers);
close all

%% Compute Joint Angles and Save
% init_hip_pos_r = fliplr(gait_trackers.jnt_pos_r(1,:,1))';
% init_knee_pos_r = fliplr(gait_trackers.jnt_pos_r(2,:,1))';
% init_ankle_pos_r = fliplr(gait_trackers.jnt_pos_r(3,:,1))';
% init_foot_pos_r = fliplr(gait_trackers.jnt_pos_r(4,:,1))';
% init_hip_pos_l = fliplr(gait_trackers.jnt_pos_l(1,:,1))';
% init_knee_pos_l = fliplr(gait_trackers.jnt_pos_l(2,:,1))';
% init_ankle_pos_l = fliplr(gait_trackers.jnt_pos_l(3,:,1))';
% init_foot_pos_l = fliplr(gait_trackers.jnt_pos_l(4,:,1))';
% 
% jnt_angles_all_r = zeros(3,3,length(t_trackers));
% jnt_angles_all_l = zeros(3,3,length(t_trackers));
% for i = 2:length(t_trackers)
%     hip_pos_r = fliplr(gait_trackers.jnt_pos_r(1,:,i))';
%     knee_pos_r = fliplr(gait_trackers.jnt_pos_r(2,:,i))';
%     ankle_pos_r = fliplr(gait_trackers.jnt_pos_r(3,:,i))';
%     foot_pos_r = fliplr(gait_trackers.jnt_pos_r(4,:,i))';
%     hip_pos_l = fliplr(gait_trackers.jnt_pos_l(1,:,i))';
%     knee_pos_l = fliplr(gait_trackers.jnt_pos_l(2,:,i))';
%     ankle_pos_l = fliplr(gait_trackers.jnt_pos_l(3,:,i))';
%     foot_pos_l = fliplr(gait_trackers.jnt_pos_l(4,:,i))';
% 
%     [hip_add_abd_r,hip_int_ext_r,hip_flex_ext_r,knee_add_abd_r,knee_int_ext_r,knee_flex_ext_r,ankle_inv_eve_r,ankle_add_abd_r,ankle_dors_plan_r] = get_jnt_angle_simplified_6dof(hip_pos_r,knee_pos_r,ankle_pos_r,foot_pos_r,init_hip_pos_r,init_knee_pos_r,init_ankle_pos_r,init_foot_pos_r);
%     [hip_add_abd_l,hip_int_ext_l,hip_flex_ext_l,knee_add_abd_l,knee_int_ext_l,knee_flex_ext_l,ankle_inv_eve_l,ankle_add_abd_l,ankle_dors_plan_l] = get_jnt_angle_simplified_6dof(hip_pos_l,knee_pos_l,ankle_pos_l,foot_pos_l,init_hip_pos_l,init_knee_pos_l,init_ankle_pos_l,init_foot_pos_l);
%     jnt_angles_r = [hip_add_abd_r,hip_int_ext_r,hip_flex_ext_r;
%         knee_add_abd_r,knee_int_ext_r,knee_flex_ext_r;
%         ankle_inv_eve_r,ankle_add_abd_r,ankle_dors_plan_r];
%     jnt_angles_l = [hip_add_abd_l,hip_int_ext_l,hip_flex_ext_l;
%         knee_add_abd_l,knee_int_ext_l,knee_flex_ext_l;
%         ankle_inv_eve_l,ankle_add_abd_l,ankle_dors_plan_l];
%     jnt_angles_all_r(:,:,i) = jnt_angles_r;
%     jnt_angles_all_l(:,:,i) = jnt_angles_l;
% end


%% Normaliza data
ankle_h_r = reshape(gait_trackers.jnt_pos_r(3,2,:),1,[]);
ankle_h_r = (ankle_h_r - mean(ankle_h_r)) / std(ankle_h_r);

ankle_h_l = reshape(gait_trackers.jnt_pos_l(3,2,:),1,[]);
ankle_h_l = (ankle_h_l - mean(ankle_h_l)) / std(ankle_h_l);

foot_h_r = reshape(gait_trackers.jnt_pos_r(4,2,:),1,[]);
foot_h_r = (foot_h_r - mean(foot_h_r)) / std(foot_h_r);

foot_h_l = reshape(gait_trackers.jnt_pos_l(4,2,:),1,[]);
foot_h_l = (foot_h_l - mean(foot_h_l)) / std(foot_h_l);

trace_r = (gait_insole.foot_trace_r - mean(gait_insole.foot_trace_r)) / std(gait_insole.foot_trace_r);
trace_l = (gait_insole.foot_trace_l - mean(gait_insole.foot_trace_l)) / std(gait_insole.foot_trace_l);

%% Plot time trace of insole on top of ankle/foot height
% t_insole_r = t_insole_r + 0.5843;
% t_insole_l = t_insole_l + 0.5971;

figure()
plot(t_trackers,ankle_h_r)
hold on
plot(t_insole_r,trace_r)
plot(t_trackers(gait_trackers.strike_r),ankle_h_r(gait_trackers.strike_r),'Marker','x','LineStyle','none','Color','green')
plot(t_insole_r(gait_insole.strike_r),trace_r(gait_insole.strike_r),'Marker','o','LineStyle','none','Color','green')
plot(t_trackers(gait_trackers.off_r),ankle_h_r(gait_trackers.off_r),'Marker','x','LineStyle','none','Color','red')
plot(t_insole_r(gait_insole.off_r),trace_r(gait_insole.off_r),'Marker','o','LineStyle','none','Color','red')
hold off
title("Time Trace of Insole and Ankle Height on Right Foot")
xlabel("Time (s)")
ylabel("Normalized Amplitude")
legend("Ankle Height","Insole")

figure()
plot(t_trackers,ankle_h_l)
hold on
plot(t_insole_l,trace_l)
plot(t_trackers(gait_trackers.strike_l),ankle_h_l(gait_trackers.strike_l),'Marker','x','LineStyle','none','Color','green')
plot(t_insole_l(gait_insole.strike_l),trace_l(gait_insole.strike_l),'Marker','o','LineStyle','none','Color','green')
plot(t_trackers(gait_trackers.off_l),ankle_h_l(gait_trackers.off_l),'Marker','x','LineStyle','none','Color','red')
plot(t_insole_l(gait_insole.off_l),trace_l(gait_insole.off_l),'Marker','o','LineStyle','none','Color','red')
hold off
title("Time Trace of Insole and Ankle Height on Left Foot")
xlabel("Time (s)")
ylabel("Normalized Amplitude")
legend("Ankle Height","Insole")

%% Analyze diff between heel strikes and toe offs detected by trackers vs insole
strike_diff_r = t_insole_r(gait_insole.strike_r(2:end-2)) - t_trackers(gait_trackers.strike_r);
off_diff_r = t_insole_r(gait_insole.off_r(1:end-1)) - t_trackers(gait_trackers.off_r);

strike_diff_l = t_insole_l(gait_insole.strike_l(1:end-3)) - t_trackers(gait_trackers.strike_l);
off_diff_l = t_insole_l(gait_insole.off_l(1:end-1)) - t_trackers(gait_trackers.off_l);

figure()
plot(strike_diff_r,'Marker','.','LineStyle','none','Color','green')
hold on
plot(off_diff_r,'Marker','.','LineStyle','none','Color','red')
hold off
title("Differences in Heel Strike and Toe off times between the Insole and Trackers for the Right Foot")

figure()
plot(strike_diff_l,'Marker','.','LineStyle','none','Color','green')
hold on
plot(off_diff_l,'Marker','.','LineStyle','none','Color','red')
hold off
title("Differences in Heel Strike and Toe off times between the Insole and Trackers for the Left Foot")

%% Plot ankle flexion angles for each step
figure()
for i = 1:length(gait_trackers.strike_l)
    idx = gait_trackers.strike_l(i):gait_trackers.off_l(i);
    pk = find(islocalmax(gait_trackers.jnt_angles_l(3,3,idx),MinProminence=2));
    plot(reshape(gait_trackers.jnt_angles_l(3,3,idx),[],1))
    hold on
    % xline(pk(1))
    % xline(gait_trackers.off_l(i) - gait_trackers.strike_l(i))
end
hold off

%% Save synced data
% save(filename,'imuAll_r','imuAll_l',"insoleAll_r","insoleAll_l","jnt_angles_all_l",'jnt_angles_all_r',"t_insole_r",'t_insole_l','t_trackers',"t_start",'jnt_pos_all_r','jnt_pos_all_l');
