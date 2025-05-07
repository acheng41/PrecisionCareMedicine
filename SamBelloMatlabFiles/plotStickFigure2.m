% Animate gait cycle given joint positions
function [] = plotStickFigure2(p,jnt_pos_r,jnt_pos_l,color_r,color_l,lim_x,lim_y,lim_z)
if iscell(jnt_pos_r)
    jnt_pos_r = cell2mat(jnt_pos_r);
end
if iscell(jnt_pos_l)
    jnt_pos_l = cell2mat(jnt_pos_l);
end

if nargin > 5
    axes(p)
    plot3(0,0,0)
    title("Vive Tracker Location")
    xlim(lim_x)
    ylim(lim_y)
    zlim(lim_z)
    xticks(floor(lim_x(1)):.5:ceil(lim_x(2)))
    yticks(floor(lim_y(1)):.5:ceil(lim_y(2)))
    zticks(floor(lim_z(1)):.5:ceil(lim_z(2)))
    xlabel("X")
    ylabel("Y")
    zlabel("Z")
    hold on
else
    axes(p)
    plot3(0,0,0)
    title("Vive Tracker Location")
    xlabel("X")
    ylabel("Y")
    zlabel("Z")
end

% plot head
% dx = 0:pi/50:2*pi;
% head_y = 2 * cos(dx) + jnt_pos(1,2);
% head_z = 4 * sin(dx) + jnt_pos(1,3) + 11.5;
% plot3(zeros(size(head_y)),head_y,head_z,'Color',color,'LineWidth',2);

% plot torso
% plot3([jnt_pos(1,1) jnt_pos(1,1)],[jnt_pos(1,2) jnt_pos(1,2)],[jnt_pos(1,3) jnt_pos(1,3) + 7.5],'Color',color,'LineWidth',2)
plot3([jnt_pos_r(1,1) jnt_pos_l(1,1)],[jnt_pos_r(1,2) jnt_pos_l(1,2)],[jnt_pos_r(1,3) jnt_pos_l(1,3)],'Color','black','LineWidth',2); % r thigh

% plot arms
% plot3([jnt_pos(1,1) jnt_pos(1,1)],[jnt_pos(1,2) jnt_pos(1,2) - 2.5],[jnt_pos(1,3) + 5 jnt_pos(1,3) - 2.5],'Color',color,'LineWidth',2)
% plot3([jnt_pos(1,1) jnt_pos(1,1)],[jnt_pos(1,2) jnt_pos(1,2) + 2.5],[jnt_pos(1,3) + 5 jnt_pos(1,3) - 2.5],'Color',color,'LineWidth',2)

% plot hip + thigh
plot3(jnt_pos_r(1,1),jnt_pos_r(1,2),jnt_pos_r(1,3),'o','LineWidth',2,'Color',color_r) % hip
plot3([jnt_pos_r(1,1) jnt_pos_r(2,1)],[jnt_pos_r(1,2) jnt_pos_r(2,2)],[jnt_pos_r(1,3) jnt_pos_r(2,3)],'Color',color_r,'LineWidth',2); % r thigh
plot3(jnt_pos_l(1,1),jnt_pos_l(1,2),jnt_pos_l(1,3),'o','LineWidth',2,'Color',color_l) % hip
plot3([jnt_pos_l(1,1) jnt_pos_l(2,1)],[jnt_pos_l(1,2) jnt_pos_l(2,2)],[jnt_pos_l(1,3) jnt_pos_l(2,3)],'Color',color_l,'LineWidth',2); % l thigh

% plot knee + shin
plot3(jnt_pos_r(2,1),jnt_pos_r(2,2),jnt_pos_r(2,3),'o','LineWidth',2,'Color',color_r) % r knee
plot3([jnt_pos_r(2,1) jnt_pos_r(3,1)],[jnt_pos_r(2,2) jnt_pos_r(3,2)],[jnt_pos_r(2,3) jnt_pos_r(3,3)],'Color',color_r,'LineWidth',2); % r shin
plot3(jnt_pos_l(2,1),jnt_pos_l(2,2),jnt_pos_l(2,3),'o','LineWidth',2,'Color',color_l) % l knee
plot3([jnt_pos_l(2,1) jnt_pos_l(3,1)],[jnt_pos_l(2,2) jnt_pos_l(3,2)],[jnt_pos_l(2,3) jnt_pos_l(3,3)],'Color',color_l,'LineWidth',2); % l shin

% plot ankle + foot
plot3(jnt_pos_r(3,1),jnt_pos_r(3,2),jnt_pos_r(3,3),'o','LineWidth',2,'Color',color_r) % r ankle
plot3(jnt_pos_r(4,1),jnt_pos_r(4,2),jnt_pos_r(4,3),'o','LineWidth',2,'Color',color_r) % r foot
plot3([jnt_pos_r(3,1) jnt_pos_r(4,1)],[jnt_pos_r(3,2) jnt_pos_r(4,2)],[jnt_pos_r(3,3) jnt_pos_r(4,3)],'Color',color_r,'LineWidth',2); % r foot
plot3(jnt_pos_l(3,1),jnt_pos_l(3,2),jnt_pos_l(3,3),'o','LineWidth',2,'Color',color_l) % l ankle
plot3(jnt_pos_l(4,1),jnt_pos_l(4,2),jnt_pos_l(4,3),'o','LineWidth',2,'Color',color_l) % l foot
plot3([jnt_pos_l(3,1) jnt_pos_l(4,1)],[jnt_pos_l(3,2) jnt_pos_l(4,2)],[jnt_pos_l(3,3) jnt_pos_l(4,3)],'Color',color_l,'LineWidth',2); % l foot
hold off

drawnow limitrate;